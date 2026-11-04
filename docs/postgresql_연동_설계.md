# 백엔드 PostgreSQL 연동 설계 초안

기준: `docs199-Server` develop 브랜치의 V3–V10 마이그레이션과 `docs/ai-server-contract.md` (확인 시점 커밋 `fb1c5b2`).

## 현재 백엔드 계약

- `documents.id`, `document_versions.id`는 UUID이며, 문서에 `current_version`과 `deleted_at`이 있다. 버전에는 `version_number`와 원본 파일의 `storage_key`가 있다.
- 원본은 PostgreSQL 본문 컬럼이 아니라 오브젝트 저장소에 저장된다. `sources.kind`는 `LOCAL`, `OBSIDIAN`, `NOTION` 등이다.
- 청크와 임베딩 테이블은 아직 없다. PostgreSQL 스키마는 백엔드 Flyway가 관리하며 `ai_app`은 기존 문서 관련 테이블에 SELECT만 허용된다.
- 백엔드는 pgvector 이미지와 Neo4j를 제공한다. 현재 AI 서버 계약은 그래프 저장소로 Neo4j를 명시한다.

## 제안하는 연결 방식

1. 백엔드 Flyway에 청크 마이그레이션을 추가한다. 각 청크는 `document_versions.id`를 외래 키로 참조하고, 청크 순번, 원문, 문자 위치, 모델 식별자와 pgvector 임베딩을 저장한다. 임베딩 차원은 실제 모델에 맞춰 확정한다.
2. PostgreSQL Full-text Search와 pgvector 검색을 각각 실행하고 RRF로 결합한다. 두 쿼리 모두 삭제되지 않은 문서와 현재 버전만 검색한다.
3. 백엔드의 `AccessScope(departmentId, position, clearance, isAdmin)` 전달과 SQL 조건이 아직 미확정이므로, 기존 데모의 문자열 그룹을 운영 권한으로 간주하지 않는다. 범위가 확정될 때까지 실제 문서 검색을 연결하지 않는다.
4. 그래프는 우선 백엔드에서 제공하는 Neo4j 계약을 유지한다. PostgreSQL 관계 테이블 또는 Apache AGE로 변경하려면 백엔드와 별도 계약 변경이 필요하다.
5. 청크 테이블의 생성과 AI 쓰기 권한은 백엔드 Flyway 마이그레이션에서 함께 정의한다. AI 프로세스가 실행 시 기존 백엔드 테이블을 만들거나 수정하지 않는다.

## 남은 결정

- 임베딩 모델과 차원, 모델 교체 시 재색인 방식
- 한국어 전문 검색의 분석기 및 검색 품질 평가
- `AccessScope`와 문서 부서·직급·보안등급을 연결하는 SQL 조건
- 원문 오프셋과 원본 파일 페이지·문단 위치의 표현, 버전 삭제 시 청크 처리
- 청크 쓰기 권한과 인덱싱 요청/콜백 계약
