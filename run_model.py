import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# 예시: 모델이 생성한 응답
predictions = [
    "FastAPI는 Python 기반의 비동기 웹 프레임워크입니다."
]

# 기대하는 정답 (정답 문장)
references = [
    "FastAPI는 비동기 Python 웹 프레임워크입니다."
]

# BLEU
bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
print("🔵 BLEU:", bleu_result["bleu"])

# ROUGE
rouge_result = rouge.compute(predictions=predictions, references=references)
print("🔴 ROUGE:", rouge_result)

# BERTScore
bert_result = bertscore.compute(predictions=predictions, references=references, lang="ko")
print("🟢 BERTScore:", {
    "precision": round(sum(bert_result["precision"]) / len(bert_result["precision"]), 4),
    "recall": round(sum(bert_result["recall"]) / len(bert_result["recall"]), 4),
    "f1": round(sum(bert_result["f1"]) / len(bert_result["f1"]), 4),
})
