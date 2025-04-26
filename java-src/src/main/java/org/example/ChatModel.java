package org.example;

import com.google.cloud.vertexai.VertexAI;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatModel;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatOptions;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

import java.util.*;

// need to set GOOGLE_APPLICATION_CREDENTIALS = "path/to/your/credentials.json"

public class ChatModel {
    private VertexAI vertexApi;
    private VertexAiGeminiChatModel chatModel;
    private final VertexAiGeminiChatOptions OPTION = VertexAiGeminiChatOptions.builder()
            .temperature(0.7)
            .topP(0.8)
            .topK(40)
            .maxOutputTokens(2048)
            .stopSequences(List.of())
            .candidateCount(1)
            .model(VertexAiGeminiChatModel.ChatModel.GEMINI_2_0_FLASH)
            .build();
            /* 1st tuning option
        VertexAiGeminiChatOptions options = VertexAiGeminiChatOptions.builder()
                .temperature(0.2)
                .topP(0.2)
                .topK(3)
                .maxOutputTokens(1000)
                .model(VertexAiGeminiChatModel.ChatModel.GEMINI_2_0_FLASH)
                .build();
        */
        /* 2nd tuning option
        VertexAiGeminiChatOptions options = VertexAiGeminiChatOptions.builder()
                .temperature((double)0.7F)
                .topP((double)1.0F)
                .topK(40)
                .maxOutputTokens(500)
                .stopSequences(List.of())
                .candidateCount(1)
                .model(VertexAiGeminiChatModel.ChatModel.GEMINI_2_0_FLASH)
                .build();
        */

    public ChatModel(String projectId, String location) {
        this.vertexApi = new VertexAI(projectId, location);
        this.chatModel = VertexAiGeminiChatModel.builder()
                .vertexAI(vertexApi)
                .defaultOptions(OPTION)
                .build();
    }

    public VertexAiGeminiChatModel getChatModel() {
        return this.chatModel;
    }

    public ChatResponse generate(String userPrompt) {
        Prompt chatPrompt = new Prompt(userPrompt);
        return this.chatModel.call(chatPrompt);
    }

    //test the chat model
    public static void main(String[] args) {
        String projectId = System.getenv("VERTEX_AI_GEMINI_PROJECT_ID");
        String location = System.getenv("VERTEX_AI_GEMINI_LOCATION");

        System.out.println("VERTEX_AI_GEMINI_PROJECT_ID: " + projectId);
        System.out.println("VERTEX_AI_GEMINI_LOCATION: " + location);

        ChatModel chatModel = new ChatModel(projectId, location);
        ChatResponse response = chatModel.generate("Can you explain Polynorpishm in OOP in detail");
        System.out.println(response);
    }

}