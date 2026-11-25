from dataclasses import dataclass, replace
from datetime import date


@dataclass(frozen=True)
class RetrievalFilter:
    document_ids: tuple[int, ...] = ()
    department: str | None = None
    product: str | None = None
    version: str | None = None
    doc_type: str | None = None
    updated_after: date | None = None
    updated_before: date | None = None

    @property
    def is_empty(self) -> bool:
        return self == EMPTY_FILTER

    @property
    def attributes_empty(self) -> bool:
        return self == replace(EMPTY_FILTER, document_ids=self.document_ids)

    @property
    def key(self) -> str:
        if self.is_empty:
            return ""
        parts = [
            f"documents={','.join(str(i) for i in sorted(set(self.document_ids)))}",
            f"department={self.department}",
            f"product={self.product}",
            f"version={self.version}",
            f"doc_type={self.doc_type}",
            f"after={self.updated_after}",
            f"before={self.updated_before}",
        ]
        return ";".join(parts)

    def merged_under(self, explicit: "RetrievalFilter") -> "RetrievalFilter":
        return RetrievalFilter(
            document_ids=explicit.document_ids or self.document_ids,
            department=explicit.department or self.department,
            product=explicit.product or self.product,
            version=explicit.version or self.version,
            doc_type=explicit.doc_type or self.doc_type,
            updated_after=explicit.updated_after or self.updated_after,
            updated_before=explicit.updated_before or self.updated_before,
        )


EMPTY_FILTER = RetrievalFilter()
