package org.example;
import org.springframework.ai.document.Document;
import java.util.*;
import java.nio.file.Path;

public class Main {
    public static void main(String[] args) {

        DocProcess enntity = new DocProcess();
        try {
            List<Document> documents = enntity.loadAndSplit(Path.of("data/TGDD/laptop.csv"), Path.of("data/TGDD/description.csv"));
            for (Document doc : documents) {
                System.out.println(doc);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}