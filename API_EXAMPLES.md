# CARELINK AI Microservice - API Examples

Complete API usage examples for the CARELINK AI Microservice.

---

## Base URL

```
http://localhost:8000
```

---

## Authentication

If API key authentication is enabled:

```bash
curl -H "X-API-Key: your-api-key-here" \
  http://localhost:8000/api/v1/triage/predict
```

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check service health and model status.

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "carelink-ai-service",
  "version": "1.0.0",
  "model_loaded": true,
  "model_version": "v1.0.0"
}
```

---

### 2. Triage Prediction

**POST** `/api/v1/triage/predict`

Main endpoint for symptom triage and risk assessment.

#### Example 1: High-Risk Case (Chest Pain)

```bash
curl -X POST http://localhost:8000/api/v1/triage/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms_text": "I have severe chest pain and sweating",
    "age": 55,
    "duration_days": 0,
    "chronic_conditions": ["diabetes", "hypertension"],
    "language": "en"
  }'
```

**Response:**

```json
{
  "prediction": "HIGH",
  "probabilities": {
    "low": 0.02,
    "medium": 0.13,
    "high": 0.85
  },
  "confidence": 0.93,
  "rules_triggered": [
    "CRITICAL_SYMPTOM_CHECK",
    "CHEST_PAIN_CARDIOVASCULAR",
    "HIGH_RISK_COMBINATION"
  ],
  "explanation": "Based on your symptoms of severe chest pain and sweating with cardiovascular risk factors, immediate medical attention is recommended. Please call emergency services (911) or go to the nearest emergency room immediately.",
  "model_version": "v1.0.0",
  "escalated": false,
  "emergency_flag": true,
  "request_id": "req_a1b2c3d4e5f6",
  "timestamp": "2026-02-27T10:30:00.000000Z"
}
```

#### Example 2: Medium-Risk Case (Fever)

```bash
curl -X POST http://localhost:8000/api/v1/triage/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms_text": "I have a fever and severe headache for 2 days",
    "age": 32,
    "duration_days": 2,
    "chronic_conditions": [],
    "language": "en"
  }'
```

**Response:**

```json
{
  "prediction": "MEDIUM",
  "probabilities": {
    "low": 0.15,
    "medium": 0.68,
    "high": 0.17
  },
  "confidence": 0.68,
  "rules_triggered": [],
  "explanation": "Based on your symptoms of fever and severe headache, you should consult a healthcare provider soon. Monitor your condition closely. Schedule an appointment with your healthcare provider within 24-48 hours.",
  "model_version": "v1.0.0",
  "escalated": false,
  "emergency_flag": false,
  "request_id": "req_b2c3d4e5f6g7",
  "timestamp": "2026-02-27T10:31:00.000000Z"
}
```

#### Example 3: Low-Risk Case (Common Cold)

```bash
curl -X POST http://localhost:8000/api/v1/triage/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms_text": "I have a runny nose and mild cough",
    "age": 28,
    "duration_days": 3,
    "chronic_conditions": [],
    "language": "en"
  }'
```

**Response:**

```json
{
  "prediction": "LOW",
  "probabilities": {
    "low": 0.82,
    "medium": 0.15,
    "high": 0.03
  },
  "confidence": 0.82,
  "rules_triggered": [],
  "explanation": "Based on your symptoms of runny nose and mild cough, your condition appears manageable. Consider consulting a healthcare provider if symptoms worsen. Self-care measures may be appropriate. Contact a healthcare provider if symptoms worsen or persist.",
  "model_version": "v1.0.0",
  "escalated": false,
  "emergency_flag": false,
  "request_id": "req_c3d4e5f6g7h8",
  "timestamp": "2026-02-27T10:32:00.000000Z"
}
```

#### Example 4: Infant with Fever (Special Case)

```bash
curl -X POST http://localhost:8000/api/v1/triage/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms_text": "My baby has a fever",
    "age": 1,
    "duration_days": 0,
    "chronic_conditions": [],
    "language": "en"
  }'
```

**Response:**

```json
{
  "prediction": "MEDIUM",
  "probabilities": {
    "low": 0.30,
    "medium": 0.55,
    "high": 0.15
  },
  "confidence": 0.75,
  "rules_triggered": ["INFANT_FEVER"],
  "explanation": "Based on your baby's fever symptoms, medical evaluation is recommended. Fever in infants requires prompt assessment. Schedule an appointment with your healthcare provider within 24-48 hours.",
  "model_version": "v1.0.0",
  "escalated": true,
  "emergency_flag": false,
  "request_id": "req_d4e5f6g7h8i9",
  "timestamp": "2026-02-27T10:33:00.000000Z"
}
```

#### Example 5: Multilingual (Spanish)

```bash
curl -X POST http://localhost:8000/api/v1/triage/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms_text": "Tengo dolor de cabeza y mareos",
    "age": 40,
    "duration_days": 1,
    "chronic_conditions": [],
    "language": "es"
  }'
```

**Response:**

```json
{
  "prediction": "MEDIUM",
  "probabilities": {
    "low": 0.25,
    "medium": 0.60,
    "high": 0.15
  },
  "confidence": 0.70,
  "rules_triggered": [],
  "explanation": "Según sus síntomas de dolor de cabeza y mareos, debe consultar a un proveedor de atención médica pronto. Monitoree su condición de cerca...",
  "model_version": "v1.0.0",
  "escalated": false,
  "emergency_flag": false,
  "request_id": "req_e5f6g7h8i9j0",
  "timestamp": "2026-02-27T10:34:00.000000Z"
}
```

---

### 3. Get Metrics

**GET** `/api/v1/metrics`

Retrieve service metrics.

```bash
curl http://localhost:8000/api/v1/metrics
```

**Response:**

```json
{
  "total_predictions": 1523,
  "predictions_by_risk": {
    "LOW": 610,
    "MEDIUM": 532,
    "HIGH": 381
  },
  "average_confidence": 0.78,
  "rules_triggered_count": 245,
  "escalations_count": 89,
  "emergency_flags_count": 127,
  "average_response_time_ms": 145.3,
  "error_count": 3,
  "uptime_seconds": 86400,
  "timestamp": "2026-02-27T10:35:00.000000Z"
}
```

---

### 4. Get Prometheus Metrics

**GET** `/api/v1/metrics/prometheus`

Export metrics in Prometheus format.

```bash
curl http://localhost:8000/api/v1/metrics/prometheus
```

**Response:**

```
# HELP carelink_ai_predictions_total Total number of predictions
# TYPE carelink_ai_predictions_total counter
carelink_ai_predictions_total 1523

# HELP carelink_ai_predictions_by_risk Predictions by risk level
# TYPE carelink_ai_predictions_by_risk counter
carelink_ai_predictions_by_risk{risk="LOW"} 610
carelink_ai_predictions_by_risk{risk="MEDIUM"} 532
carelink_ai_predictions_by_risk{risk="HIGH"} 381

# HELP carelink_ai_average_confidence Average prediction confidence
# TYPE carelink_ai_average_confidence gauge
carelink_ai_average_confidence 0.78
...
```

---

### 5. Get Model Info

**GET** `/api/v1/model/info`

Retrieve model information and metadata.

```bash
curl http://localhost:8000/api/v1/model/info
```

**Response:**

```json
{
  "model_version": "v1.0.0",
  "metadata": {
    "version": "v1.0.0",
    "created_at": "2026-02-27T00:00:00Z",
    "training_date": "2026-02-20T00:00:00Z",
    "model_type": "RandomForestClassifier",
    "performance": {
      "accuracy": 0.89,
      "precision": {
        "low": 0.91,
        "medium": 0.87,
        "high": 0.90
      }
    }
  },
  "loaded_at": "2026-02-27T08:00:00.000000Z"
}
```

---

### 6. Get Active Rules

**GET** `/api/v1/rules`

List all active clinical safety rules.

```bash
curl http://localhost:8000/api/v1/rules
```

**Response:**

```json
{
  "rules_enabled": true,
  "rules": [
    {
      "name": "CRITICAL_SYMPTOM_CHECK",
      "description": "Checks for presence of critical symptoms",
      "enabled": true
    },
    {
      "name": "HIGH_RISK_COMBINATION",
      "description": "Detects high-risk symptom combinations",
      "enabled": true
    },
    {
      "name": "CHEST_PAIN_CARDIOVASCULAR",
      "description": "Evaluates chest pain with cardiovascular risk factors",
      "enabled": true
    },
    {
      "name": "PREGNANCY_COMPLICATIONS",
      "description": "Identifies pregnancy-related warning signs",
      "enabled": true
    }
  ],
  "version": "1.0.0"
}
```

---

## Python Client Example

```python
import requests

class CarelinkAIClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.api_key = api_key
    
    def predict(self, symptoms_text, age, duration_days, 
                chronic_conditions=None, language="en"):
        """Make a triage prediction."""
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        payload = {
            "symptoms_text": symptoms_text,
            "age": age,
            "duration_days": duration_days,
            "chronic_conditions": chronic_conditions or [],
            "language": language
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/triage/predict",
            json=payload,
            headers=headers
        )
        
        response.raise_for_status()
        return response.json()

# Usage
client = CarelinkAIClient(api_key="your-api-key")

result = client.predict(
    symptoms_text="I have chest pain and sweating",
    age=52,
    duration_days=1,
    chronic_conditions=["diabetes"]
)

print(f"Risk Level: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Explanation: {result['explanation']}")
```

---

## JavaScript/TypeScript Client Example

```typescript
interface TriageRequest {
  symptoms_text: string;
  age: number;
  duration_days: number;
  chronic_conditions?: string[];
  language?: string;
}

interface TriageResponse {
  prediction: 'LOW' | 'MEDIUM' | 'HIGH';
  probabilities: {
    low: number;
    medium: number;
    high: number;
  };
  confidence: number;
  rules_triggered: string[];
  explanation: string;
  model_version: string;
  escalated: boolean;
  emergency_flag: boolean;
  request_id: string;
  timestamp: string;
}

class CarelinkAIClient {
  constructor(
    private baseUrl: string = 'http://localhost:8000',
    private apiKey?: string
  ) {}

  async predict(request: TriageRequest): Promise<TriageResponse> {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const response = await fetch(
      `${this.baseUrl}/api/v1/triage/predict`,
      {
        method: 'POST',
        headers,
        body: JSON.stringify(request),
      }
    );

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  }
}

// Usage
const client = new CarelinkAIClient('http://localhost:8000', 'your-api-key');

const result = await client.predict({
  symptoms_text: 'I have chest pain and sweating',
  age: 52,
  duration_days: 1,
  chronic_conditions: ['diabetes'],
  language: 'en',
});

console.log(`Risk Level: ${result.prediction}`);
console.log(`Confidence: ${result.confidence}`);
console.log(`Explanation: ${result.explanation}`);
```

---

## Error Handling

### 400 Bad Request

Invalid input data:

```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "ensure this value is less than or equal to 150",
      "type": "value_error.number.not_le"
    }
  ]
}
```

### 401 Unauthorized

Missing or invalid API key:

```json
{
  "detail": "Invalid API key"
}
```

### 429 Too Many Requests

Rate limit exceeded:

```json
{
  "detail": "Rate limit exceeded. Please try again later."
}
```

### 500 Internal Server Error

Server error:

```json
{
  "detail": "An error occurred while processing your request. Please try again."
}
```

---

## Rate Limiting

Default: 60 requests per minute per IP

To increase limits, configure in `.env`:

```bash
RATE_LIMIT_PER_MINUTE=120
```

---

## Best Practices

1. **Always handle errors**: Implement proper error handling in your client
2. **Use request IDs**: Store request_id for debugging and audit trails
3. **Respect rate limits**: Implement exponential backoff
4. **Secure API keys**: Never commit API keys to version control
5. **Validate responses**: Check response structure before using data
6. **Monitor usage**: Track your API calls and performance
7. **Keep dependencies updated**: Regularly update client libraries

---

For more information, visit the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
