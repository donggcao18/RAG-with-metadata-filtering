package org.example;

import io.qdrant.client.QdrantClient;
import io.qdrant.client.QdrantGrpcClient;
import io.qdrant.client.grpc.Collections.Distance;
import io.qdrant.client.grpc.Collections.VectorParams;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.qdrant.QdrantVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.ai.document.Document;
import org.springframework.context.annotation.Bean;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutionException;


public class VectorDatabase {
    private VectorStore vectorStore;

    public QdrantClient createQdrantClient(String StoreName) throws ExecutionException, InterruptedException {
        String hostname = "localhost";
        int port = 6334;
        boolean isTls = false;
        String apiKey = System.getenv("QDRANT_API_KEY");

        QdrantGrpcClient.Builder grpcClientBuilder = QdrantGrpcClient.newBuilder(hostname, port, isTls);

        if (apiKey != null && !apiKey.isEmpty()) {
            grpcClientBuilder.withApiKey(apiKey);
        }
        QdrantClient client = new QdrantClient(grpcClientBuilder.build());

        try {
            client.createCollectionAsync(StoreName,
                    VectorParams.newBuilder()
                            .setDistance(Distance.Cosine)
                            .setSize(768)
                            .build()).get();
        } catch (Exception e) {
            System.out.println("Collection may already exist: " + e.getMessage());
        }
        return client;
    }

    public VectorStore vectorStore(QdrantClient qdrantClient,
                                   EmbeddingModel embeddingModel,
                                   String StoreName){

        return QdrantVectorStore.builder(qdrantClient, embeddingModel)
                .collectionName(StoreName)
                .initializeSchema(true)
                .batchingStrategy(new TokenCountBatchingStrategy())
                .build();
    }

    public VectorDatabase(String StoreName) throws IOException, ExecutionException, InterruptedException {
        QdrantClient qClient = createQdrantClient(StoreName);
        EmbedModel eModel = new EmbedModel();
        this.vectorStore = vectorStore(qClient, eModel.getEmbeddingModel(), StoreName);
    }

    public void addToQdrant(List<Document> documents) {
        final int BATCH_SIZE = 1;

        for (int i = 0; i < documents.size(); i += BATCH_SIZE) {
            int endIndex = Math.min(i + BATCH_SIZE, documents.size());
            List<Document> batch = documents.subList(i, endIndex);

            try {
                vectorStore.add(batch);
                System.out.println("Embedded " + i + " data points");

                Thread.sleep(5000);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                System.out.println("Sleep interrupted: " + e.getMessage());
                break;
            } catch (Exception e) {
                System.out.println("Failed to embed batch " + i + "-" + endIndex + ": " + e.getMessage());
                continue;
            }
        }
    }

    public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
        String storeName = "laptop";
        VectorDatabase vectorStore = new VectorDatabase(storeName);

        List<Document> documents = List.of(
                new Document("Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!!", Map.of("meta1", "meta1")),
                new Document("The World is Big and Salvation Lurks Around the Corner"),
                new Document("You walk forward facing the past and you turn back toward the future.", Map.of("meta2", "meta2")));
        vectorStore.addToQdrant(documents);
    }
}