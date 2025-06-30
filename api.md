# AI Agent System - API Documentation & SDKs

## 📚 RESTful API Documentation

### Base URL
```
Production: https://api.ai-agents.example.com/v1
Staging: https://staging-api.ai-agents.example.com/v1
```

### Authentication
All API requests require authentication using JWT tokens or API keys.

```http
Authorization: Bearer <your-jwt-token>
# or
X-API-Key: <your-api-key>
```

## 🔧 Core API Endpoints

### Document Processing

#### Submit Document for Processing
```http
POST /documents/process

Request:
{
  "document_id": "invoice_2024_001.pdf",
  "document_url": "https://storage.example.com/documents/invoice_2024_001.pdf",
  "priority": 5,
  "metadata": {
    "document_type": "invoice",
    "source": "email",
    "quarter": "Q1-2024"
  },
  "processing_options": {
    "extraction_strategy": "auto",
    "enable_ocr": true,
    "language": "en",
    "output_format": "structured_json"
  }
}

Response:
{
  "request_id": "req_a1b2c3d4e5f6",
  "status": "processing",
  "estimated_completion": "2024-03-15T10:30:00Z",
  "workflow_id": "wf_123456",
  "tracking_url": "https://api.ai-agents.example.com/v1/status/req_a1b2c3d4e5f6"
}
```

#### Get Processing Status
```http
GET /status/{request_id}

Response:
{
  "request_id": "req_a1b2c3d4e5f6",
  "status": "completed",
  "progress": 100,
  "stages": {
    "extraction": {
      "status": "completed",
      "duration_ms": 823,
      "agent_id": "extractor_002"
    },
    "analysis": {
      "status": "completed",
      "duration_ms": 456,
      "anomaly_score": 0.12,
      "agent_id": "analyzer_001"
    },
    "validation": {
      "status": "completed",
      "duration_ms": 234,
      "confidence": 0.96
    }
  },
  "result": {
    "extraction_id": "ext_789xyz",
    "download_url": "https://api.ai-agents.example.com/v1/results/ext_789xyz",
    "expires_at": "2024-03-16T10:30:00Z"
  }
}
```

#### Get Extraction Results
```http
GET /results/{extraction_id}

Response:
{
  "extraction_id": "ext_789xyz",
  "document_id": "invoice_2024_001.pdf",
  "extracted_data": {
    "invoice_number": "INV-2024-001",
    "date": "2024-03-01",
    "vendor": {
      "name": "ACME Corporation",
      "address": "123 Main St, City, State 12345",
      "tax_id": "12-3456789"
    },
    "line_items": [
      {
        "description": "Professional Services",
        "quantity": 40,
        "unit_price": 150.00,
        "total": 6000.00
      }
    ],
    "total_amount": 6480.00,
    "tax_amount": 480.00,
    "currency": "USD"
  },
  "confidence_scores": {
    "overall": 0.96,
    "fields": {
      "invoice_number": 0.99,
      "date": 0.98,
      "vendor.name": 0.95,
      "total_amount": 0.97
    }
  },
  "metadata": {
    "processing_time_ms": 1513,
    "model_version": "v2.1.0",
    "extracted_at": "2024-03-15T10:28:45Z"
  }
}
```

### Conversational Interface

#### Start Conversation
```http
POST /chat/sessions

Request:
{
  "user_id": "user_123",
  "context": {
    "organization_id": "org_456",
    "preferred_language": "en"
  }
}

Response:
{
  "session_id": "chat_abc123",
  "created_at": "2024-03-15T10:00:00Z",
  "expires_at": "2024-03-15T18:00:00Z"
}
```

#### Send Message
```http
POST /chat/sessions/{session_id}/messages

Request:
{
  "message": "I need to process an invoice from our supplier",
  "attachments": [
    {
      "type": "document",
      "url": "https://storage.example.com/invoice.pdf"
    }
  ]
}

Response:
{
  "message_id": "msg_123",
  "response": "I'll help you process that invoice. I've detected it's from ACME Corporation dated March 1st, 2024. Would you like me to extract all the details and line items?",
  "suggested_actions": [
    {
      "action": "extract_all",
      "label": "Yes, extract everything"
    },
    {
      "action": "extract_summary",
      "label": "Just give me the summary"
    }
  ],
  "intent": {
    "type": "document_processing",
    "confidence": 0.98
  }
}
```

### Feedback & Learning

#### Submit Feedback
```http
POST /feedback

Request:
{
  "extraction_id": "ext_789xyz",
  "feedback_type": "correction",
  "corrections": {
    "vendor.name": {
      "original": "ACME Corporation",
      "corrected": "ACME Corp International",
      "reason": "Full legal name required"
    }
  },
  "quality_rating": 4,
  "comments": "Generally accurate but missed the full company name"
}

Response:
{
  "feedback_id": "fb_456",
  "status": "accepted",
  "impact": {
    "immediate": "Document re-processed with corrections",
    "learning": "Model will be updated in next training cycle",
    "similar_documents": 15
  }
}
```

### Analytics & Insights

#### Get Performance Metrics
```http
GET /analytics/performance?period=last_30_days

Response:
{
  "period": {
    "start": "2024-02-15",
    "end": "2024-03-15"
  },
  "metrics": {
    "total_documents": 15234,
    "success_rate": 0.965,
    "average_processing_time_ms": 1847,
    "anomalies_detected": 423,
    "human_interventions": 89
  },
  "trends": {
    "success_rate_change": "+2.3%",
    "processing_time_change": "-15.2%",
    "automation_rate": 0.942
  },
  "top_document_types": [
    {"type": "invoice", "count": 8234, "percentage": 0.54},
    {"type": "contract", "count": 4123, "percentage": 0.27},
    {"type": "report", "count": 2877, "percentage": 0.19}
  ]
}
```

## 🌐 Multi-Language SDKs

### Python SDK

```python
# Installation
pip install ai-agent-sdk

# Usage Example
from ai_agent_sdk import AIAgentClient
from ai_agent_sdk.models import ProcessingOptions, DocumentType

# Initialize client
client = AIAgentClient(
    api_key="your-api-key",
    base_url="https://api.ai-agents.example.com/v1"
)

# Process document
async def process_invoice():
    # Submit document
    result = await client.documents.process(
        document_path="invoice.pdf",
        document_type=DocumentType.INVOICE,
        options=ProcessingOptions(
            enable_ocr=True,
            extraction_strategy="advanced",
            output_format="structured_json"
        )
    )
    
    # Wait for completion
    extraction = await client.wait_for_completion(
        result.request_id,
        timeout=300,
        poll_interval=2
    )
    
    # Get results
    data = extraction.extracted_data
    print(f"Invoice Number: {data.invoice_number}")
    print(f"Total Amount: {data.total_amount}")
    
    # Submit feedback if needed
    if data.vendor.name != "ACME Corp International":
        await client.feedback.submit(
            extraction_id=extraction.extraction_id,
            corrections={
                "vendor.name": {
                    "original": data.vendor.name,
                    "corrected": "ACME Corp International"
                }
            }
        )

# Conversational interface
async def chat_example():
    # Start session
    session = await client.chat.create_session()
    
    # Send message
    response = await client.chat.send_message(
        session_id=session.session_id,
        message="Show me all invoices processed today"
    )
    
    print(response.message)
    
    # Stream responses
    async for chunk in client.chat.stream_message(
        session_id=session.session_id,
        message="Analyze trends in our Q1 documents"
    ):
        print(chunk.content, end="")

# Batch processing
async def batch_example():
    documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    
    # Process batch with progress tracking
    async with client.batch_processor() as batch:
        for doc in documents:
            await batch.add_document(doc)
        
        async for progress in batch.process():
            print(f"Progress: {progress.completed}/{progress.total}")
            if progress.failed:
                print(f"Failed: {progress.failed_documents}")
```

### TypeScript/JavaScript SDK

```typescript
// Installation
npm install @ai-agents/sdk

// Usage Example
import { AIAgentClient, DocumentType, ProcessingOptions } from '@ai-agents/sdk';

// Initialize client
const client = new AIAgentClient({
  apiKey: process.env.AI_AGENT_API_KEY,
  baseUrl: 'https://api.ai-agents.example.com/v1'
});

// Process document with type safety
async function processDocument() {
  try {
    // Submit document
    const result = await client.documents.process({
      documentUrl: 'https://storage.example.com/invoice.pdf',
      documentType: DocumentType.Invoice,
      options: {
        enableOcr: true,
        extractionStrategy: 'advanced',
        outputFormat: 'structured_json'
      } as ProcessingOptions
    });
    
    // Poll for status
    const extraction = await client.waitForCompletion(result.requestId, {
      timeout: 300000, // 5 minutes
      pollInterval: 2000
    });
    
    // Type-safe access to results
    console.log(`Invoice Number: ${extraction.extractedData.invoiceNumber}`);
    console.log(`Total: ${extraction.extractedData.totalAmount}`);
    
    // React hook example
    const { data, isLoading, error } = useDocumentExtraction(result.requestId);
    
  } catch (error) {
    if (error instanceof AIAgentError) {
      console.error(`API Error: ${error.code} - ${error.message}`);
    }
  }
}

// Real-time streaming with WebSocket
const stream = client.streaming.connect();

stream.on('connected', () => {
  console.log('Connected to AI Agent stream');
});

stream.on('document.processed', (event) => {
  console.log(`Document ${event.documentId} processed`);
});

stream.on('anomaly.detected', (event) => {
  console.log(`Anomaly detected: ${event.severity}`);
});

// React Component Example
import { AIAgentProvider, useAIAgent } from '@ai-agents/react';

function App() {
  return (
    <AIAgentProvider apiKey={process.env.AI_AGENT_API_KEY}>
      <DocumentProcessor />
    </AIAgentProvider>
  );
}

function DocumentProcessor() {
  const { processDocument, isProcessing } = useAIAgent();
  
  const handleUpload = async (file: File) => {
    const result = await processDocument(file, {
      type: DocumentType.Invoice,
      priority: 5
    });
    
    console.log('Extraction:', result);
  };
  
  return (
    <div>
      <input type="file" onChange={(e) => handleUpload(e.target.files[0])} />
      {isProcessing && <div>Processing...</div>}
    </div>
  );
}
```

### Go SDK

```go
// Installation
go get github.com/ai-agents/go-sdk

// Usage Example
package main

import (
    "context"
    "fmt"
    "log"
    
    aiagent "github.com/ai-agents/go-sdk"
)

func main() {
    // Initialize client
    client := aiagent.NewClient(
        aiagent.WithAPIKey("your-api-key"),
        aiagent.WithBaseURL("https://api.ai-agents.example.com/v1"),
    )
    
    ctx := context.Background()
    
    // Process document
    result, err := client.Documents.Process(ctx, &aiagent.ProcessRequest{
        DocumentID: "invoice.pdf",
        DocumentURL: "https://storage.example.com/invoice.pdf",
        Options: &aiagent.ProcessingOptions{
            EnableOCR: true,
            ExtractionStrategy: "advanced",
        },
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Wait for completion
    extraction, err := client.WaitForCompletion(ctx, result.RequestID,
        aiagent.WithTimeout(5*time.Minute),
        aiagent.WithPollInterval(2*time.Second),
    )
    if err != nil {
        log.Fatal(err)
    }
    
    // Access results
    fmt.Printf("Invoice Number: %s\n", extraction.ExtractedData.InvoiceNumber)
    fmt.Printf("Total Amount: %.2f\n", extraction.ExtractedData.TotalAmount)
    
    // Concurrent batch processing
    documents := []string{"doc1.pdf", "doc2.pdf", "doc3.pdf"}
    results := make(chan *aiagent.ExtractionResult, len(documents))
    
    var wg sync.WaitGroup
    for _, doc := range documents {
        wg.Add(1)
        go func(documentID string) {
            defer wg.Done()
            
            res, _ := client.Documents.Process(ctx, &aiagent.ProcessRequest{
                DocumentID: documentID,
            })
            
            extraction, _ := client.WaitForCompletion(ctx, res.RequestID)
            results <- extraction
        }(doc)
    }
    
    wg.Wait()
    close(results)
    
    // Process results
    for result := range results {
        fmt.Printf("Processed: %s\n", result.DocumentID)
    }
}
```

### Java SDK

```java
// Installation - Maven
<dependency>
    <groupId>com.aiagents</groupId>
    <artifactId>ai-agent-sdk</artifactId>
    <version>1.0.0</version>
</dependency>

// Usage Example
import com.aiagents.sdk.*;
import com.aiagents.sdk.models.*;

public class AIAgentExample {
    public static void main(String[] args) {
        // Initialize client
        AIAgentClient client = AIAgentClient.builder()
            .apiKey("your-api-key")
            .baseUrl("https://api.ai-agents.example.com/v1")
            .build();
        
        try {
            // Process document
            ProcessingOptions options = ProcessingOptions.builder()
                .enableOcr(true)
                .extractionStrategy(ExtractionStrategy.ADVANCED)
                .outputFormat(OutputFormat.STRUCTURED_JSON)
                .build();
            
            ProcessingResult result = client.documents().process(
                "invoice.pdf",
                DocumentType.INVOICE,
                options
            );
            
            // Wait for completion with progress callback
            Extraction extraction = client.waitForCompletion(
                result.getRequestId(),
                Duration.ofMinutes(5),
                progress -> System.out.println("Progress: " + progress.getPercentage() + "%")
            );
            
            // Access results
            InvoiceData data = extraction.getExtractedData(InvoiceData.class);
            System.out.println("Invoice Number: " + data.getInvoiceNumber());
            System.out.println("Total Amount: " + data.getTotalAmount());
            
            // Async processing with CompletableFuture
            CompletableFuture<Extraction> future = client.documents()
                .processAsync("document.pdf")
                .thenCompose(res -> client.waitForCompletionAsync(res.getRequestId()));
            
            future.thenAccept(ext -> {
                System.out.println("Async processing completed: " + ext.getExtractionId());
            });
            
            // Reactive streams with RxJava
            client.documents()
                .processReactive("document.pdf")
                .flatMap(res -> client.getStatusStream(res.getRequestId()))
                .filter(status -> status.getProgress() > 50)
                .subscribe(
                    status -> System.out.println("Progress: " + status.getProgress()),
                    error -> System.err.println("Error: " + error),
                    () -> System.out.println("Processing completed")
                );
                
        } catch (AIAgentException e) {
            System.err.println("API Error: " + e.getCode() + " - " + e.getMessage());
        }
    }
}
```

### C# / .NET SDK

```csharp
// Installation - NuGet
Install-Package AIAgent.SDK

// Usage Example
using AIAgent.SDK;
using AIAgent.SDK.Models;

public class Program
{
    static async Task Main(string[] args)
    {
        // Initialize client
        var client = new AIAgentClient(new AIAgentOptions
        {
            ApiKey = "your-api-key",
            BaseUrl = "https://api.ai-agents.example.com/v1"
        });
        
        try
        {
            // Process document
            var result = await client.Documents.ProcessAsync(new ProcessRequest
            {
                DocumentId = "invoice.pdf",
                DocumentUrl = "https://storage.example.com/invoice.pdf",
                Options = new ProcessingOptions
                {
                    EnableOcr = true,
                    ExtractionStrategy = ExtractionStrategy.Advanced,
                    OutputFormat = OutputFormat.StructuredJson
                }
            });
            
            // Wait for completion with cancellation
            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(5));
            var extraction = await client.WaitForCompletionAsync(
                result.RequestId,
                pollInterval: TimeSpan.FromSeconds(2),
                cancellationToken: cts.Token
            );
            
            // Strongly typed results
            var invoiceData = extraction.ExtractedData.ToObject<InvoiceData>();
            Console.WriteLine($"Invoice Number: {invoiceData.InvoiceNumber}");
            Console.WriteLine($"Total Amount: {invoiceData.TotalAmount:C}");
            
            // LINQ queries on results
            var highValueItems = invoiceData.LineItems
                .Where(item => item.Total > 1000)
                .OrderByDescending(item => item.Total)
                .ToList();
            
            // Async enumerable for batch processing
            var documents = new[] { "doc1.pdf", "doc2.pdf", "doc3.pdf" };
            
            await foreach (var doc in ProcessDocumentsAsync(client, documents))
            {
                Console.WriteLine($"Processed: {doc.DocumentId}");
            }
        }
        catch (AIAgentException ex)
        {
            Console.WriteLine($"Error: {ex.ErrorCode} - {ex.Message}");
        }
    }
    
    static async IAsyncEnumerable<Extraction> ProcessDocumentsAsync(
        AIAgentClient client,
        string[] documents)
    {
        var tasks = documents.Select(doc => 
            client.Documents.ProcessAsync(new ProcessRequest { DocumentId = doc })
        ).ToList();
        
        foreach (var task in tasks)
        {
            var result = await task;
            var extraction = await client.WaitForCompletionAsync(result.RequestId);
            yield return extraction;
        }
    }
}

// ASP.NET Core Integration
public class Startup
{
    public void ConfigureServices(IServiceCollection services)
    {
        services.AddAIAgentClient(options =>
        {
            options.ApiKey = Configuration["AIAgent:ApiKey"];
            options.BaseUrl = Configuration["AIAgent:BaseUrl"];
        });
    }
}

[ApiController]
[Route("api/[controller]")]
public class DocumentsController : ControllerBase
{
    private readonly IAIAgentClient _aiAgent;
    
    public DocumentsController(IAIAgentClient aiAgent)
    {
        _aiAgent = aiAgent;
    }
    
    [HttpPost("process")]
    public async Task<IActionResult> ProcessDocument(IFormFile file)
    {
        var result = await _aiAgent.Documents.ProcessAsync(file.OpenReadStream());
        return Ok(result);
    }
}
```

### Ruby SDK

```ruby
# Installation
gem install ai_agent_sdk

# Usage Example
require 'ai_agent_sdk'

# Initialize client
client = AIAgent::Client.new(
  api_key: ENV['AI_AGENT_API_KEY'],
  base_url: 'https://api.ai-agents.example.com/v1'
)

# Process document
result = client.documents.process(
  document_id: 'invoice.pdf',
  document_url: 'https://storage.example.com/invoice.pdf',
  options: {
    enable_ocr: true,
    extraction_strategy: 'advanced'
  }
)

# Wait for completion
extraction = client.wait_for_completion(
  result.request_id,
  timeout: 300,
  poll_interval: 2
)

# Access results
puts "Invoice Number: #{extraction.extracted_data['invoice_number']}"
puts "Total Amount: #{extraction.extracted_data['total_amount']}"

# Batch processing with parallel execution
documents = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']

results = Parallel.map(documents, in_threads: 3) do |doc|
  res = client.documents.process(document_id: doc)
  client.wait_for_completion(res.request_id)
end

results.each do |extraction|
  puts "Processed: #{extraction.document_id}"
end

# Rails Integration
class DocumentsController < ApplicationController
  before_action :initialize_ai_client
  
  def process_document
    result = @ai_client.documents.process(
      document_id: params[:file].original_filename,
      document_data: params[:file].read
    )
    
    ProcessDocumentJob.perform_later(result.request_id)
    
    render json: { request_id: result.request_id }
  end
  
  private
  
  def initialize_ai_client
    @ai_client = AIAgent::Client.new(
      api_key: Rails.application.credentials.ai_agent_api_key
    )
  end
end
```

## 🔐 Authentication & Security

### API Key Management
```python
# Rotate API keys programmatically
from ai_agent_sdk import AIAgentClient

client = AIAgentClient(api_key="current-key")

# Generate new API key
new_key = client.auth.rotate_api_key(
    current_key="current-key",
    expires_in_days=90
)

# Update client with new key
client.update_api_key(new_key.key)
```

### OAuth 2.0 Flow
```typescript
// OAuth authentication
const auth = new AIAgentAuth({
  clientId: process.env.CLIENT_ID,
  clientSecret: process.env.CLIENT_SECRET,
  redirectUri: 'https://app.example.com/callback'
});

// Get authorization URL
const authUrl = auth.getAuthorizationUrl({
  scope: ['documents:read', 'documents:write'],
  state: 'random-state'
});

// Exchange code for token
const token = await auth.exchangeCodeForToken(code);

// Use token with client
const client = new AIAgentClient({
  accessToken: token.access_token
});
```

## 📊 Rate Limiting & Quotas

| Plan | Requests/Minute | Concurrent Requests | Monthly Documents |
|------|----------------|--------------------|--------------------|
| Free | 10 | 2 | 100 |
| Starter | 60 | 10 | 1,000 |
| Professional | 300 | 50 | 10,000 |
| Enterprise | Unlimited | Unlimited | Unlimited |

### Handling Rate Limits
```python
# SDK handles rate limiting automatically
client = AIAgentClient(
    api_key="your-key",
    retry_config={
        "max_retries": 3,
        "backoff_factor": 2,
        "rate_limit_retry": True
    }
)

# Manual rate limit handling
try:
    result = await client.documents.process(document)
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after} seconds")
    await asyncio.sleep(e.retry_after)
    result = await client.documents.process(document)
```

## 🧪 Testing & Development

### Mock Server
```bash
# Start mock server for development
docker run -p 8080:8080 aiagents/mock-server:latest

# Configure SDK to use mock server
export AI_AGENT_BASE_URL=http://localhost:8080
```

### Integration Testing
```python
import pytest
from ai_agent_sdk import AIAgentClient
from ai_agent_sdk.testing import MockAIAgentServer

@pytest.fixture
def mock_server():
    with MockAIAgentServer() as server:
        yield server

def test_document_processing(mock_server):
    # Configure mock response
    mock_server.add_response(
        "POST", "/documents/process",
        json={"request_id": "test_123", "status": "processing"}
    )
    
    # Test with mock
    client = AIAgentClient(base_url=mock_server.url)
    result = client.documents.process("test.pdf")
    
    assert result.request_id == "test_123"
```

## 📱 Mobile SDKs

### iOS (Swift)
```swift
// Installation - Swift Package Manager
dependencies: [
    .package(url: "https://github.com/ai-agents/swift-sdk.git", from: "1.0.0")
]

// Usage
import AIAgentSDK

let client = AIAgentClient(apiKey: "your-api-key")

// Process document from camera
let image = // captured image
let result = try await client.documents.process(
    image: image,
    documentType: .invoice,
    options: ProcessingOptions(enableOCR: true)
)

// SwiftUI Integration
struct DocumentScannerView: View {
    @StateObject private var scanner = DocumentScanner()
    
    var body: some View {
        VStack {
            if let extraction = scanner.extraction {
                ExtractionResultView(extraction: extraction)
            } else {
                CameraScannerView(onCapture: scanner.processDocument)
            }
        }
    }
}
```

### Android (Kotlin)
```kotlin
// Installation - Gradle
implementation 'com.aiagents:android-sdk:1.0.0'

// Usage
import com.aiagents.sdk.AIAgentClient
import com.aiagents.sdk.models.*

class DocumentActivity : AppCompatActivity() {
    private val client = AIAgentClient(apiKey = "your-api-key")
    
    suspend fun processDocument(uri: Uri) {
        val result = client.documents.process(
            documentUri = uri,
            documentType = DocumentType.INVOICE,
            options = ProcessingOptions(
                enableOcr = true,
                extractionStrategy = ExtractionStrategy.ADVANCED
            )
        )
        
        // Wait for completion with coroutines
        val extraction = withContext(Dispatchers.IO) {
            client.waitForCompletion(result.requestId)
        }
        
        // Update UI
        withContext(Dispatchers.Main) {
            displayResults(extraction)
        }
    }
}
```

## 🌍 Global Edge Deployment

### Edge Locations
- North America: `us-east-1`, `us-west-2`, `ca-central-1`
- Europe: `eu-west-1`, `eu-central-1`, `eu-north-1`
- Asia Pacific: `ap-southeast-1`, `ap-northeast-1`, `ap-south-1`
- South America: `sa-east-1`

### Latency Optimization
```python
# SDK automatically routes to nearest edge
client = AIAgentClient(
    api_key="your-key",
    edge_routing=True,  # Enable automatic edge routing
    preferred_region="us-west-2"  # Optional preferred region
)

# Get latency metrics
metrics = client.get_edge_metrics()
print(f"Nearest edge: {metrics.nearest_edge}")
print(f"Latency: {metrics.latency_ms}ms")
```

## 📞 Support

- **Documentation**: [https://docs.ai-agents.example.com](https://docs.ai-agents.example.com)
- **API Status**: [https://status.ai-agents.example.com](https://status.ai-agents.example.com)
- **Community Forum**: [https://community.ai-agents.example.com](https://community.ai-agents.example.com)
- **Enterprise Support**: support@ai-agents.example.com