package com.example.ragchat;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.beans.factory.annotation.Value;

@SpringBootApplication
public class RagChatApplication {

    public static void main(String[] args) {
        SpringApplication.run(RagChatApplication.class, args);
    }

    @Bean
    public WebClient ragWebClient(@Value("${rag.service.url}") String baseUrl) {
        return WebClient.builder().baseUrl(baseUrl).build();
    }
}
