# 1-team-FC-ai




## 🚀 YouTil-AI GitHub 협업 가이드

이 프로젝트는 두 주요 모듈을 포함합니다:
- **Lydia**: Interview 생성 기능
- **Kai**: TIL(Today I Learned) 자동 작성 기능, 테크 news 요약 기능

각 모듈은 `dev/interview`, `dev/til` 브랜치에서 개발되며, 이 문서는 협업을 위한 Git 규칙, 이슈 관리, 커밋 메시지 작성, PR 절차를 명시합니다.

---

## 🌱 브랜치 전략

- `main`: 운영 및 배포용 브랜치 (직접 push 금지)
- `dev`: 기능 통합 개발 브랜치
- `dev/interview`: Lydia 인터뷰 기능 개발 브랜치
- `dev/til`: Kai TIL 기능 개발 브랜치
- `feature/{lydia|kai}/이슈번호-작업명`: 기능 개발 브랜치
- `bugfix/{lydia|kai}/이슈번호-수정내용`: 버그 수정 브랜치

> 예시:
> - `feature/lydia/23-interview-generator`
> - `bugfix/kai/17-fix-til-parsing`

---

## 🗂️ 작업 흐름

1. **이슈 등록**
   - GitHub Issue 생성 후 브랜치 작업 시작
   - 모듈(interview/til) 라벨 지정
2. **브랜치 생성**
   - 규칙에 맞는 브랜치명 사용
3. **작업 및 커밋**
   - 커밋 메시지 규칙 준수
4. **PR(Pull Request) 생성 및 리뷰**
   - `dev/interview` 또는 `dev/til`로 PR 요청
   - 최소 1명 이상의 코드 리뷰 승인 후 병합

---

## 🐛 이슈 템플릿

```markdown
### 📌 작업 개요
- 작업 명: (예: 인터뷰 질문 생성 API 개선)

### ✅ 작업 내용
- [ ] 세부 작업 항목 1
- [ ] 세부 작업 항목 2

### 🔁 관련 이슈
- 관련된 이슈 번호: #이슈번호

### 💬 기타 참고사항
- API 명세 변경 있음 / 프론트와 연동 예정 등
```

---

### 💬 커밋 메시지 규칙
커밋 메시지는 다음 형식을 따릅니다:

```markdown
[type][#이슈번호] 작업 요약
```

- `feat[#23] 인터뷰 질문 생성 API 구현`
- `fix[#17] TIL JSON 파싱 오류 수정`
- `docs[#30] README 작업 흐름 정리`

| 타입     | 설명                    |
|----------|-------------------------|
| feat     | 새로운 기능             |
| fix      | 버그 수정               |
| refactor | 리팩토링                |
| docs     | 문서 추가/수정          |
| test     | 테스트 추가/수정        |
| chore    | 설정, 빌드 등 기타 작업  |

### 🔀 Pull Request 규칙
- PR 제목 형식
```markdown
[#이슈번호] 작업 요약
```
- Base 브랜치
  - `dev/interview` 또는 `dev/til`

- **PR 내용 포함 항목**
  - 작업 개요
  - 주요 변경사항
  - 테스트 결과 (필요시 캡처)
  - 관련 이슈 링크
- **리뷰 및 병합**
  - Self-merge 시 리뷰 코멘트 필수
