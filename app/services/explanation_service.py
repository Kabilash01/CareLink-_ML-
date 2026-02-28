from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import List, Optional

from app.nlp.language_handler import LanguageHandler
from app.core.logging import logger
from app.core.settings import get_settings

settings = get_settings()


class ExplanationService:
    """
    MedGemma 4B-IT — single NF4 4-bit model for BOTH text and vision.

    All GPU operations run in a **dedicated worker thread** that also loads
    the model.  BitsAndBytes NF4 dequantisation uses thread-local CUDA state;
    running generate() from a different thread triggers
    ``TensorCompare.cu  assertion `input[0] != 0` failed``.
    The worker pattern keeps every CUDA call on the same OS thread.
    """

    def __init__(self):
        self.enabled = settings.EXPLANATION_SERVICE_ENABLED
        self.language_handler = LanguageHandler()

        self._model = None          # single NF4 model — text AND vision
        self._processor = None      # Gemma3Processor (text + image)
        self._tokenizer = None      # processor.tokenizer alias

        self._loading = False
        self._load_err = None

        # GPU worker queue: callers submit (callable, result_event, result_box)
        self._gpu_queue: queue.Queue = queue.Queue()

        if self.enabled:
            self._start_load()

    # ─────────────────────────────────────────────
    # GPU WORKER — load model then serve inference
    # ─────────────────────────────────────────────

    def _start_load(self):
        self._loading = True
        t = threading.Thread(target=self._gpu_worker, daemon=True)
        t.start()
        logger.info(f"MedGemma GPU worker started | model={settings.LLM_HF_MODEL_ID}")

    def _dispatch_to_gpu(self, fn, *args, **kwargs):
        """Submit *fn* to the GPU worker thread and block until it returns."""
        result_box: list = []
        done_event = threading.Event()

        self._gpu_queue.put((fn, args, kwargs, result_box, done_event))
        done_event.wait()  # blocks caller until worker finishes

        if len(result_box) == 2 and result_box[0] == "__exc__":
            raise result_box[1]
        return result_box[0]

    def _gpu_worker(self):
        """Load the model, run warmups, then serve inference requests forever."""
        self._load_model()

        # ── Enter request loop (runs until process exits)
        while True:
            try:
                fn, args, kwargs, result_box, done_event = self._gpu_queue.get()
                try:
                    result_box.append(fn(*args, **kwargs))
                except Exception as exc:
                    result_box.extend(["__exc__", exc])
                finally:
                    done_event.set()
            except Exception:
                pass  # keep the worker alive

    def _load_model(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

            # ── SSL fix: huggingface_hub 1.x uses httpx; patch its client factory
            import httpx
            import urllib3
            import huggingface_hub.utils._http as _hf_http
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            _hf_http.close_session()
            _hf_http.set_client_factory(
                lambda: httpx.Client(
                    verify=False, follow_redirects=True,
                    timeout=httpx.Timeout(120.0),
                )
            )
            _hf_http.get_session()
            os.environ["CURL_CA_BUNDLE"] = ""
            os.environ["REQUESTS_CA_BUNDLE"] = ""
            logger.info("[BNB] SSL verification bypassed (httpx client patched)")

            # ── CUDA check
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available — NF4 requires a CUDA GPU")
            logger.info(f"[BNB] CUDA OK | device={torch.cuda.get_device_name(0)}")

            # ── TF32 for faster matmul on Ampere+
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            model_id = settings.LLM_HF_MODEL_ID
            hf_token = settings.HUGGINGFACE_TOKEN or None

            # ── Processor (text + image)
            logger.info(f"[BNB] Loading processor: {model_id}")
            self._processor = AutoProcessor.from_pretrained(
                model_id, token=hf_token, padding_side="right", use_fast=True,
            )
            self._tokenizer = self._processor.tokenizer
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # ── Single NF4 4-bit model
            logger.info(f"[BNB] Loading model (NF4 4-bit): {model_id}")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            t0 = time.time()
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, device_map="auto",
                dtype=torch.bfloat16, token=hf_token, low_cpu_mem_usage=True,
            )
            self._model.eval()
            load_ms = (time.time() - t0) * 1000

            devices = {str(p.device) for p in self._model.parameters()}
            logger.info(f"[BNB] Parameter devices: {devices}")

            # ── Warmup: text
            logger.info("[BNB] Warmup (text)...")
            enc = self._tokenizer("ping", return_tensors="pt")
            with torch.no_grad():
                self._model.generate(
                    enc.input_ids.to("cuda"),
                    attention_mask=enc.attention_mask.to("cuda"),
                    max_new_tokens=1, use_cache=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            logger.info("[BNB] Text warmup OK")

            # ── Warmup: vision
            logger.info("[BNB] Warmup (vision)...")
            try:
                import numpy as np
                from PIL import Image as _PIL
                _img = _PIL.fromarray(np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8))
                _msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "ping"}]}]
                _txt = self._processor.apply_chat_template(_msg, add_generation_prompt=True, tokenize=False)
                _inp = self._processor(text=_txt, images=[_img], return_tensors="pt", padding=False)
                _safe = {k: v for k, v in _inp.items() if k != "token_type_ids"}
                if "pixel_values" in _safe and _safe["pixel_values"].dtype != torch.bfloat16:
                    _safe["pixel_values"] = _safe["pixel_values"].to(torch.bfloat16)
                _safe = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in _safe.items()}
                torch.cuda.empty_cache()
                with torch.no_grad():
                    _out = self._model.generate(
                        **_safe, max_new_tokens=10,
                        min_new_tokens=5,
                        do_sample=False,           # NF4+vision+sampling → CUDA assert on Blackwell
                        repetition_penalty=settings.LLM_REPEAT_PENALTY,
                        use_cache=True, pad_token_id=self._tokenizer.eos_token_id,
                    )
                _ntok = _out[0].shape[0] - _safe["input_ids"].shape[-1]
                logger.info(f"[BNB] Vision warmup OK | generated {_ntok} tokens")
                del _img, _inp, _safe
                torch.cuda.empty_cache()
            except Exception as ve:
                logger.warning(f"[BNB] Vision warmup failed: {ve}")

            logger.info(f"[BNB] Ready | model={model_id} load={load_ms:.0f}ms nf4=True TF32=True")
            self._loading = False

        except Exception as exc:
            self._load_err = str(exc)
            self._loading = False
            logger.error(f"MedGemma load failed: {exc} — template fallback active")

    # ─────────────────────────────────────────────
    # TEXT INFERENCE (runs in GPU worker thread)
    # ─────────────────────────────────────────────

    def _text_infer(self, prompt: str, max_new_tokens: int = 200) -> str:
        return self._dispatch_to_gpu(self._do_text_infer, prompt, max_new_tokens)

    def _do_text_infer(self, prompt: str, max_new_tokens: int) -> str:
        import torch
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=settings.LLM_MAX_LENGTH,
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to("cuda")

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
                repetition_penalty=settings.LLM_REPEAT_PENALTY,
                use_cache=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        logger.info(f"[BNB] text generate() | {new_ids.shape[0]} tok | {elapsed:.2f}s")
        return text

    # ─────────────────────────────────────────────
    # VISION INFERENCE (runs in GPU worker thread)
    # ─────────────────────────────────────────────

    def _vision_infer(self, image, question: str, max_new_tokens: int = 200) -> str:
        return self._dispatch_to_gpu(self._do_vision_infer, image, question, max_new_tokens)

    def _do_vision_infer(self, image, question: str, max_new_tokens: int) -> str:
        """
        Two-step Gemma3 pattern on the GPU worker thread:
          1. apply_chat_template(tokenize=False) → text with <image> placeholder
          2. processor(text, images) → input_ids + attention_mask + pixel_values
        Filter out token_type_ids, cast pixel_values to float16.
        Suppress pad-token generation and detect NaN logits (NF4 + Blackwell).
        """
        import torch
        from transformers import LogitsProcessorList, LogitsProcessor

        # ── Custom logits processor: suppress <pad> (id 0) and fix NaN ──
        class _SanitizeLogits(LogitsProcessor):
            """Set pad-token logit to -inf and replace any NaN with -inf."""
            def __init__(self, pad_id: int):
                self.pad_id = pad_id
                self.nan_count = 0

            def __call__(self, input_ids, scores):
                if torch.isnan(scores).any():
                    self.nan_count += 1
                    scores = torch.nan_to_num(scores, nan=-float("inf"))
                scores[:, self.pad_id] = -float("inf")
                return scores

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }]
        text_prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        inputs = self._processor(
            text=text_prompt, images=[image],
            return_tensors="pt", padding=False,
        )

        # Filter out token_type_ids, cast pixel_values, move to GPU
        safe = {k: v for k, v in inputs.items() if k != "token_type_ids"}
        if "pixel_values" in safe and safe["pixel_values"].dtype != torch.bfloat16:
            safe["pixel_values"] = safe["pixel_values"].to(torch.bfloat16)
        safe = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in safe.items()}
        input_len = safe["input_ids"].shape[-1]

        pad_id = self._tokenizer.pad_token_id or 0
        sanitizer = _SanitizeLogits(pad_id)

        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        with torch.no_grad():
            # NF4 + vision KV-cache + multinomial sampling (do_sample=True)
            # triggers TensorCompare.cu assertion on Blackwell GPUs (RTX 50xx)
            # at ANY temperature. Must use greedy decoding.
            output_ids = self._model.generate(
                **safe,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=settings.LLM_REPEAT_PENALTY,
                use_cache=True,
                pad_token_id=self._tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([sanitizer]),
            )
        elapsed = time.perf_counter() - t0

        new_ids = output_ids[0][input_len:]
        raw_text = self._tokenizer.decode(new_ids, skip_special_tokens=False)
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        logger.info(
            f"[BNB] vision generate() | {new_ids.shape[0]} tok | {elapsed:.2f}s "
            f"| raw_len={len(raw_text)} clean_len={len(text)} "
            f"| first_5_ids={new_ids[:5].tolist()} "
            f"| nan_steps={sanitizer.nan_count} "
            f"| raw_preview={raw_text[:200]!r}"
        )
        return text

    # ─────────────────────────────────────────────
    # HELPER: messages → prompt text
    # ─────────────────────────────────────────────

    def _messages_to_prompt(self, messages: list) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)

    # ─────────────────────────────────────────────
    # PUBLIC API — generate_explanation
    # ─────────────────────────────────────────────

    def generate_explanation(
        self,
        risk_level: str,
        symptoms: List[str],
        confidence: float,
        rules_triggered: List[str],
        emergency_flag: bool,
        language: str = "en",
        image=None,
    ) -> str:
        """Generate a patient-friendly explanation; MedGemma NF4 or template fallback."""
        if not self.enabled:
            return self._template_explanation(risk_level, symptoms, language)

        try:
            if self._model is not None:
                return self._llm_explanation(
                    risk_level, symptoms, confidence,
                    rules_triggered, emergency_flag, language, image=image,
                )
            if self._loading:
                logger.debug("MedGemma still loading — template fallback")
            elif self._load_err:
                logger.debug(f"MedGemma unavailable ({self._load_err}) — template fallback")
            return self._template_explanation(risk_level, symptoms, language)
        except Exception as exc:
            logger.error(f"Explanation generation error: {exc}")
            return self._template_explanation(risk_level, symptoms, language)

    # ─────────────────────────────────────────────
    # PUBLIC API — analyze_image
    # ─────────────────────────────────────────────

    def analyze_image(
        self,
        image=None,
        question: str = "What do you observe in this medical image?",
        language: str = "en",
    ) -> str:
        """
        Analyze a medical image using MedGemma's native vision encoder.
        """
        if not self.enabled:
            return "Image analysis requires the LLM service. Enable EXPLANATION_SERVICE_ENABLED in .env."

        if self._model is None:
            if self._loading:
                return "MedGemma is still loading. Please try again in a moment."
            return f"MedGemma not available: {self._load_err or 'unknown error'}"

        try:
            lang_instruction = {
                "en": "Respond in English.",
                "es": "Responde completamente en español.",
                "fr": "Réponds entièrement en français.",
            }.get(language, "Respond in English.")

            if image is not None:
                vision_prompt = (
                    "You are a compassionate and thorough medical AI assistant. "
                    "Analyze this medical image step by step:\n"
                    "1. Describe what you observe (color, texture, shape, size, location).\n"
                    "2. List possible clinical findings or differential diagnoses.\n"
                    "3. Recommend whether the patient should seek in-person evaluation.\n\n"
                    f"Patient's question: {question}\n\n"
                    f"{lang_instruction}\n"
                    "Provide a detailed response."
                )
                text = self._vision_infer(image, vision_prompt, settings.LLM_MAX_NEW_TOKENS)
            else:
                messages = [
                    {"role": "system", "content": (
                        "You are a compassionate medical AI assistant. "
                        "Describe what a clinician would look for and recommend "
                        "in-person professional evaluation."
                    )},
                    {"role": "user", "content": f"{question} {lang_instruction}"},
                ]
                text = self._text_infer(
                    self._messages_to_prompt(messages),
                    settings.LLM_MAX_NEW_TOKENS,
                )

            if not text:
                return "MedGemma returned an empty response. Please try again."
            logger.debug(f"MedGemma image analysis OK | vision={'yes' if image else 'no'} | {len(text)} chars")
            return text
        except Exception as exc:
            logger.error(f"MedGemma image analysis failed: {exc}")
            return f"Image analysis failed: {exc}"

    # ─────────────────────────────────────────────
    # PUBLIC API — predict_triage
    # ─────────────────────────────────────────────

    def predict_triage(
        self,
        symptoms_text: str,
        age: int,
        duration_days: int,
        chronic_conditions: List[str],
        language: str = "en",
    ) -> Optional[dict]:
        """Use the LLM to classify triage risk. Returns dict or None."""
        import re as _re

        if not self.enabled or self._model is None:
            return None

        try:
            chronic_str = ", ".join(chronic_conditions) if chronic_conditions else "none"
            prompt_user = (
                "You are a clinical triage AI. Based on the following patient information, "
                "determine the triage risk level.\n\n"
                f"Patient age: {age} years\n"
                f"Symptom duration: {duration_days} day(s)\n"
                f"Chronic conditions: {chronic_str}\n"
                f"Symptoms: {symptoms_text}\n\n"
                "Respond in this EXACT JSON format and nothing else:\n"
                '{\n  "risk_level": "HIGH",\n  "confidence": 0.85,\n  "reasoning": "one sentence"\n}\n\n'
                "Risk level criteria:\n"
                "  HIGH   — life-threatening, needs immediate emergency care\n"
                "  MEDIUM — urgent, needs attention within 24 hours\n"
                "  LOW    — non-urgent, can be managed at home\n"
            )

            messages = [
                {"role": "system", "content": "You are a clinical triage AI. Respond ONLY with JSON."},
                {"role": "user", "content": prompt_user},
            ]
            raw = self._text_infer(self._messages_to_prompt(messages), settings.LLM_MAX_NEW_TOKENS)

            # Extract JSON
            m = _re.search(r'\{[^}]+\}', raw, _re.DOTALL)
            if not m:
                logger.warning(f"MedGemma triage: no JSON found in: {raw[:200]}")
                return None

            data = json.loads(m.group())
            risk = data.get("risk_level", "MEDIUM").upper()
            if risk not in ("LOW", "MEDIUM", "HIGH"):
                risk = "MEDIUM"
            conf = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")

            logger.info(f"MedGemma triage → {risk} ({conf:.2f}): {reasoning[:80]}")
            return {"risk_level": risk, "confidence": conf, "reasoning": reasoning}

        except Exception as exc:
            logger.error(f"MedGemma triage prediction failed: {exc}")
            return None

    # ─────────────────────────────────────────────
    # PUBLIC API — generate_disclaimer
    # ─────────────────────────────────────────────

    def generate_disclaimer(self, language: str = "en") -> str:
        disclaimers = {
            "en": (
                "⚠️ IMPORTANT: This is an AI-generated assessment for informational purposes only. "
                "It is NOT a medical diagnosis. Always consult a qualified healthcare professional "
                "for medical advice, diagnosis, or treatment. In case of emergency, call your local "
                "emergency services immediately."
            ),
            "es": (
                "⚠️ IMPORTANTE: Esta es una evaluación generada por IA solo con fines informativos. "
                "NO es un diagnóstico médico. Siempre consulte a un profesional de la salud calificado."
            ),
            "fr": (
                "⚠️ IMPORTANT: Ceci est une évaluation générée par IA à des fins d'information uniquement. "
                "Ce n'est PAS un diagnostic médical. Consultez toujours un professionnel de santé qualifié."
            ),
        }
        return disclaimers.get(language, disclaimers["en"])

    @property
    def model_ready(self):
        return self._model is not None

    # ─────────────────────────────────────────────
    # PRIVATE — LLM explanation (model available)
    # ─────────────────────────────────────────────

    def _llm_explanation(
        self,
        risk_level: str,
        symptoms: List[str],
        confidence: float,
        rules_triggered: List[str],
        emergency_flag: bool,
        language: str = "en",
        image=None,
    ) -> str:
        lang_instruction = {
            "en": "Respond in English.",
            "es": "Responde completamente en español.",
            "fr": "Réponds entièrement en français.",
        }.get(language, "Respond in English.")

        symptoms_str = ", ".join(symptoms) if symptoms else "none reported"
        rules_str = ", ".join(rules_triggered) if rules_triggered else "none"
        emergency_note = " This is flagged as an EMERGENCY." if emergency_flag else ""

        messages = [
            {"role": "system", "content": (
                "You are a compassionate medical AI assistant. Explain triage results "
                "in simple, reassuring language. Never diagnose. Always recommend "
                "professional medical consultation."
            )},
            {"role": "user", "content": (
                f"Risk level: {risk_level}\n"
                f"Symptoms: {symptoms_str}\n"
                f"Confidence: {confidence:.0%}\n"
                f"Safety rules triggered: {rules_str}\n"
                f"{emergency_note}\n\n"
                "Provide a clear, patient-friendly explanation of what this means "
                f"and what the patient should do next. {lang_instruction}"
            )},
        ]

        text = self._text_infer(
            self._messages_to_prompt(messages),
            settings.LLM_MAX_NEW_TOKENS,
        )
        return text or self._template_explanation(risk_level, symptoms, language)

    # ─────────────────────────────────────────────
    # PRIVATE — template fallback (no model)
    # ─────────────────────────────────────────────

    def _template_explanation(self, risk_level: str, symptoms: List[str], language: str = "en") -> str:
        symptom_str = ", ".join(symptoms) if symptoms else "reported symptoms"
        templates = {
            "en": {
                "HIGH": f"⚠️ HIGH RISK: Based on {symptom_str}, immediate medical attention is strongly recommended. Please visit an emergency room or call emergency services right away.",
                "MEDIUM": f"🔶 MEDIUM RISK: Based on {symptom_str}, we recommend scheduling a medical appointment within the next 24 hours. Monitor your symptoms carefully.",
                "LOW": f"🟢 LOW RISK: Based on {symptom_str}, your symptoms appear manageable with home care. Rest, stay hydrated, and consult a doctor if symptoms worsen.",
            },
            "es": {
                "HIGH": f"⚠️ RIESGO ALTO: Basado en {symptom_str}, se recomienda atención médica inmediata.",
                "MEDIUM": f"🔶 RIESGO MEDIO: Basado en {symptom_str}, recomendamos una cita médica dentro de las próximas 24 horas.",
                "LOW": f"🟢 RIESGO BAJO: Basado en {symptom_str}, sus síntomas parecen manejables con cuidados en el hogar.",
            },
            "fr": {
                "HIGH": f"⚠️ RISQUE ÉLEVÉ: Sur la base de {symptom_str}, une attention médicale immédiate est recommandée.",
                "MEDIUM": f"🔶 RISQUE MOYEN: Sur la base de {symptom_str}, nous recommandons un rendez-vous médical dans les 24 heures.",
                "LOW": f"🟢 RISQUE FAIBLE: Sur la base de {symptom_str}, vos symptômes semblent gérables avec des soins à domicile.",
            },
        }
        lang_templates = templates.get(language, templates["en"])
        return lang_templates.get(risk_level, lang_templates.get("MEDIUM", "Please consult a doctor."))