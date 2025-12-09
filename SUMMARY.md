# Entity Identification Pipeline - Summary

## 1. Approach Explanation

### Problem Statement
Build an entity identification system for a co-pilot agent that predicts relevant entities (CDR, Phone, Web Activity, etc.) from natural language user queries.

### Data Sources
- **`user_queries.csv`**: User questions with JSON containing field references and entity labels
- **`fields_description.csv`**: Human-readable descriptions for each field (e.g., `ifc.ootb.CDR.technology` → "Technology used in communication: 2G, 3G, 4G...")

### Data Extraction from JSON

**Entity Labels Extraction:**
From each JSON in `user_queries.csv`, we extract entity labels from two keys:
- `entityType`: The primary entity (e.g., "CDR", "Phone")
- `relationTargetType`: Related entities in nested statements (e.g., when a CDR query references Phone entities)

```
JSON: {'entityType': 'CDR', 'statements': [{'type': 'relation', 'parameters': {'relationTargetType': ['Phone']}}]}
→ Extracted entities: ['CDR', 'Phone']
```

**Field Descriptions Extraction:**
For each query, we extract the `name` fields from the JSON and look up their descriptions in `fields_description.csv`:
```
Query: "Find all calls made using 3G technology"
JSON fields: ifc.ootb.CDR.technology, ifc.ootb.CDR.type
→ Descriptions: "Technology used in communication...", "Type of communication..."
```

### Two Approaches Compared

| Aspect | Pre-trained (Retrieval) | Fine-tuned (Augmented) |
|--------|------------------------|------------------------|
| **Method** | k-NN similarity search | Classification with data augmentation |
| **Training** | None (uses embeddings as-is) | Train on queries + descriptions |
| **Database** | Embed (query + descriptions) | N queries + N description entries = 2N samples |
| **Inference** | Find most similar, return its entities | Direct classification |
| **Model** | DistilBERT (frozen) | DistilBERT (fine-tuned) |

### Justification for Choices

1. **DistilBERT**: Small, fast, runs locally - suitable for co-pilot use case
2. **Description Augmentation**: Teaches the model that field descriptions map to entities, improving generalization
3. **Multi-label Classification**: Queries can reference multiple entities (20 multi-entity cases in test)
4. **80/20 Split**: Standard split with strict separation to prevent data leakage

### Evaluation Metrics

| Metric | Why Used |
|--------|----------|
| **Exact Match** | Strictest - all predicted entities must exactly match ground truth |
| **F1 Micro** | Overall performance across all predictions |
| **F1 Macro** | Balanced performance across entity types (handles class imbalance) |
| **Precision** | How many predictions are correct (avoid false positives) |
| **Recall** | How many true entities are found (avoid false negatives) |
| **Hamming Loss** | Fraction of incorrectly predicted labels |

### Results Summary

| Metric | Pre-trained | Fine-tuned | Δ |
|--------|-------------|------------|---|
| Exact Match | 59.7% | **98.0%** | +38.3% |
| F1 Micro | 69.3% | **98.8%** | +29.5% |
| Multi-entity Accuracy | - | **90.0%** | - |

---

## 2. Open Issues and Suggestions for Improvement

### Open Issues

1. **Potentially Overfit Results**
   - 98% exact match is unusually high
   - Test queries may be too similar to training queries
   - Small dataset (149 test samples) may not represent real-world diversity

2. **Multi-entity Cases**
   - Only 20/149 (13%) test cases have multiple entities
   - 90% accuracy on multi-entity is good but based on small sample

3. **Class Imbalance**
   - Some entities have few samples (EVisa Request: 2-4 samples)
   - Performance on rare entities may not be reliable

4. **Description Dependency**
   - Model learned from entity-specific descriptions
   - May not generalize to completely new query patterns

5. **Entity Order Uncertainty**
   - Current implementation treats entities as an unordered set
   - Open question: Does the order of entities matter for the downstream task?
   - If order matters (e.g., primary entity first), the evaluation and model would need adjustment
   - Current metrics (F1, Exact Match) ignore order - they only check set equality

### Suggestions for Further Improvement

1. **Cross-Validation**
   ```python
   # Use k-fold CV instead of single split
   from sklearn.model_selection import StratifiedKFold
   ```
   - More robust performance estimate
   - Better use of limited data

2. **Hard Negative Mining**
   - Add confusing examples where similar queries map to different entities
   - Improve discrimination between similar entity types

3. **Test on Novel Queries**
   - Create completely new query patterns not in training
   - True test of generalization

4. **Ensemble Approach**
   - Combine retrieval + fine-tuned predictions
   - Use retrieval as fallback for low-confidence predictions

5. **Threshold Tuning**
   - Current threshold: 0.5
   - Tune per-entity thresholds for better precision/recall tradeoff

6. **More Training Data**
   - Augment with paraphrased queries
   - Generate synthetic queries using LLM

7. **Production Monitoring**
   - Log predictions and user feedback
   - Continuously improve with real-world data
