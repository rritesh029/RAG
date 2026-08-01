package com.example.ragchat;

import java.time.Duration;
import java.util.List;
import java.util.Map;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class ChatController {

    private final WebClient ragWebClient;

    public ChatController(WebClient ragWebClient) {
        this.ragWebClient = ragWebClient;
    }

    public static record ChatRequest(String prompt, Integer k) {}

    public static record RetrievedDoc(String team, String content) {}

    public static record ChatResponse(String query, List<RetrievedDoc> results) {}

    @PostMapping(value = "/chat", consumes = MediaType.APPLICATION_JSON_VALUE,
                 produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<?> chat(@RequestBody ChatRequest req) {
        if (req == null || req.prompt() == null || req.prompt().isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("error", "prompt is required"));
        }
        int k = req.k() == null ? 3 : req.k();

        try {
            ChatResponse resp = ragWebClient.post()
                    .uri("/rag/query")
                    .bodyValue(Map.of("query", req.prompt(), "k", k))
                    .retrieve()
                    .bodyToMono(ChatResponse.class)
                    .timeout(Duration.ofSeconds(45))
                    .block();
            return ResponseEntity.ok(resp);
        } catch (Exception e) {
            return ResponseEntity.status(502)
                    .body(Map.of("error", "RAG service error: " + e.getMessage()));
        }
    }
}
