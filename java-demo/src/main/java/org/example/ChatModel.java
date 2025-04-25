package org.example;

import com.google.cloud.vertexai.VertexAI;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatModel;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatOptions;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

import java.util.*;

// need to set GOOGLE_APPLICATION_CREDENTIALS = "path/to/your/credentials.json"

public class ChatModel {
    private final VertexAI vertexApi;
    private final VertexAiGeminiChatModel chatModel;

    public ChatModel(String projectId, String location) {
        this.vertexApi = new VertexAI(projectId, location);

        /*
        VertexAiGeminiChatOptions options = VertexAiGeminiChatOptions.builder()
                .temperature(0.2)
                .topP(0.2)
                .topK(3)
                .maxOutputTokens(1000)
                .model(VertexAiGeminiChatModel.ChatModel.GEMINI_2_0_FLASH)
                .build();
        */
        VertexAiGeminiChatOptions options = VertexAiGeminiChatOptions.builder()
                .temperature(0.7)
                .topP(0.8)
                .topK(40)
                .maxOutputTokens(2048)
                .stopSequences(List.of())
                .candidateCount(1)
                .model(VertexAiGeminiChatModel.ChatModel.GEMINI_2_0_FLASH)
                .build();

        this.chatModel = VertexAiGeminiChatModel.builder()
                .vertexAI(vertexApi)
                .defaultOptions(options)
                .build();
    }

    public VertexAiGeminiChatModel getChatModel() {
        return this.chatModel;
    }

    public ChatResponse call(String userPrompt) {
        Prompt chatPrompt = new Prompt(userPrompt);
        return this.chatModel.call(chatPrompt);
    }

    public static void main(String[] args) {
        String projectId = System.getenv("VERTEX_AI_GEMINI_PROJECT_ID");
        String location = System.getenv("VERTEX_AI_GEMINI_LOCATION");

        System.out.println("GOOGLE_CLOUD_PROJECT_ID: " + projectId);
        System.out.println("GOOGLE_CLOUD_LOCATION: " + location);

        ChatModel chatModel = new ChatModel(projectId, location);
        ChatResponse response = chatModel.call("Generate the names of 5 famous national presidents.");
        System.out.println(response);
    }

}