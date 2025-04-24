package org.example;

import com.google.cloud.vertexai.VertexAI;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatModel;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatOptions;

public class ChatModel {
    private final VertexAI vertexApi;
    private final VertexAiGeminiChatModel chatModel;

    public ChatModel(String projectId, String location) {
        this.vertexApi = new VertexAI(projectId, location);

        VertexAiGeminiChatOptions options = VertexAiGeminiChatOptions.builder()
                .temperature(0.2)
                .topP(0.2)
                .topK(3)
                .maxOutputTokens(1000)
                .model(VertexAiGeminiChatModel.ChatModel.GEMINI_2_0_FLASH)
                .build();

        this.chatModel =  VertexAiGeminiChatModel.builder()
                	.vertexAI(vertexApi)
                    .vertexAiGeminiChatOptions(options)
                    .build();
    }
}