package org.example;

import java.io.FileInputStream;
import com.google.api.gax.core.FixedCredentialsProvider;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.aiplatform.v1.PredictionServiceSettings;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.vertexai.embedding.VertexAiEmbeddingConnectionDetails;
import org.springframework.ai.vertexai.embedding.text.VertexAiTextEmbeddingModel;
import org.springframework.ai.vertexai.embedding.text.VertexAiTextEmbeddingOptions;


import java.io.IOException;
import java.util.List;

public class EmbedModel {
    private final VertexAiTextEmbeddingModel embeddingModel;

    public EmbedModel() throws IOException {
        // Load Google credentials from environment variable
        String credentialsJson = System.getenv("GOOGLE_APPLICATION_CREDENTIALS");

        GoogleCredentials credentials = GoogleCredentials
                .fromStream(new FileInputStream(System.getenv("GOOGLE_APPLICATION_CREDENTIALS")))
                .createScoped("https://www.googleapis.com/auth/cloud-platform");

        credentials.refreshIfExpired();

        String endpoint = System.getenv("VERTEX_AI_ENDPOINT");
        if (endpoint == null) {
            endpoint = "us-central1-aiplatform.googleapis.com";
        }

        VertexAiEmbeddingConnectionDetails connectionDetails = VertexAiEmbeddingConnectionDetails.builder()
                .projectId(System.getenv("VERTEX_AI_GEMINI_PROJECT_ID"))
                .location(System.getenv("VERTEX_AI_GEMINI_LOCATION"))
                .apiEndpoint(endpoint)
                .predictionServiceSettings(
                        PredictionServiceSettings.newBuilder()
                                .setEndpoint(endpoint)
                                .setCredentialsProvider(FixedCredentialsProvider.create(credentials))
                                .build()
                )
                .build();

        VertexAiTextEmbeddingOptions options = VertexAiTextEmbeddingOptions.builder()
                .model("text-multilingual-embedding-002")
                .build();

        this.embeddingModel = new VertexAiTextEmbeddingModel(connectionDetails, options);
    }

    public VertexAiTextEmbeddingModel getEmbeddingModel() {
        return this.embeddingModel;
    }

    public EmbeddingResponse getEmbeddings(List<String> texts) {
        return this.embeddingModel.embedForResponse(texts);
    }
}