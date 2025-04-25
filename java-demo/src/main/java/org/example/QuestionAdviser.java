package org.example;

import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatResponse;

import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatModel;
import org.springframework.ai.vectorstore.VectorStore;

public class QuestionAdviser {
    private VertexAiGeminiChatModel chatModel;
    private VectorStore vectorStore;

    public QuestionAdviser() {
        this.chatModel = chatModel;
    }
    QuestionAnswerAdvisor qaAdvisor = new QuestionAnswerAdvisor(this.vectorStore,
            SearchRequest.builder().
                    similarityThreshold(0.8d).
                    topK(6).
                    build());
    String userText = "What is the best laptop for gaming?";
    ChatResponse response = ChatClient.builder(chatModel)
            .build().prompt()
            .advisors(new QuestionAnswerAdvisor(vectorStore))
            .user(userText)
            .call()
            .chatResponse();
}