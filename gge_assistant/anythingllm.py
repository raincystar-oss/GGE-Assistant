from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx


class AnythingLLMError(RuntimeError):
    pass


class AnythingLLMNeedsAttention(AnythingLLMError):
    pass


@dataclass(frozen=True)
class LocatedDocument:
    document_id: str | None
    title: str
    location: str
    name: str


class AnythingLLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key_path: Path,
        workspace_slug: str,
        timeout_seconds: float = 180.0,
        max_attempts: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_path = api_key_path
        self.workspace_slug = workspace_slug
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self._failed = False

    def __enter__(self) -> "AnythingLLMClient":
        api_key = self.api_key_path.read_text(encoding="utf-8").strip()
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.timeout_seconds,
            trust_env=False,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._client.close()

    def ensure_workspace(self, name: str, prompt: str) -> None:
        workspaces = self._request("GET", "/workspaces").json().get("workspaces", [])
        if not any(item.get("slug") == self.workspace_slug for item in workspaces):
            payload = {
                "name": name,
                "openAiPrompt": prompt,
                "chatProvider": "deepseek",
                "chatModel": "deepseek-v4-pro",
                "openAiTemp": 0.1,
                "openAiHistory": 0,
                "queryRefusalResponse": "Недостаточно данных в загруженных источниках.",
                "chatMode": "query",
                "similarityThreshold": 0.2,
                "topN": 8,
            }
            created = self._request("POST", "/workspace/new", json=payload).json()
            slug = created.get("workspace", {}).get("slug")
            if slug != self.workspace_slug:
                raise AnythingLLMNeedsAttention(
                    f"Workspace created with unexpected slug {slug!r}; expected {self.workspace_slug!r}"
                )
        self._request(
            "POST",
            f"/workspace/{self.workspace_slug}/update",
            json={
                "openAiPrompt": prompt,
                "chatProvider": "deepseek",
                "chatModel": "deepseek-v4-pro",
                "openAiTemp": 0.1,
                "openAiHistory": 0,
                "queryRefusalResponse": "Недостаточно данных в загруженных источниках.",
                "chatMode": "query",
                "similarityThreshold": 0.2,
                "topN": 8,
            },
        )

    def upload_or_locate_document(
        self,
        *,
        title: str,
        text: str,
        sha256: str,
        url: str,
        published: int | None,
        lose_response_for_test: bool = False,
    ) -> LocatedDocument:
        found = self.locate_document(title=title, sha256=sha256)
        if found is not None:
            return found

        body = {
            "textContent": text,
            "metadata": {
                "title": title,
                "url": url,
                "docSource": "GGE Assistant chunk",
                "chunkSource": f"{title}.txt",
                "description": f"sha256:{sha256}",
            },
        }
        if published is not None:
            body["metadata"]["published"] = published

        if lose_response_for_test:
            with self._stream("POST", "/document/raw-text", json=body) as response:
                response.raise_for_status()
        else:
            payload = self._request("POST", "/document/raw-text", json=body).json()
            if not payload.get("success"):
                raise AnythingLLMError(f"Upload failed: {payload.get('error')}")
        found = self.locate_document(title=title, sha256=sha256)
        if found is None:
            raise AnythingLLMNeedsAttention(
                f"Upload result for {title} is not uniquely recoverable; inspect /documents before retry"
            )
        return found

    def locate_document(self, *, title: str, sha256: str) -> LocatedDocument | None:
        payload = self._request("GET", "/documents").json()
        matches = [
            item
            for item in flatten_documents(payload.get("localFiles", {}))
            if item.get("title") == f"{title}.txt" and item.get("description") == f"sha256:{sha256}"
        ]
        if len(matches) > 1:
            raise AnythingLLMNeedsAttention(
                f"Multiple AnythingLLM documents match {title}/sha256:{sha256}"
            )
        if not matches:
            return None
        item = matches[0]
        self._request("GET", "/document/" + quote(item["name"], safe=""))
        return LocatedDocument(
            document_id=item.get("id"),
            title=item.get("title") or f"{title}.txt",
            location=item["location"],
            name=item["name"],
        )

    def ensure_workspace_document(self, location: str) -> str:
        workspace = self.workspace()
        doc = workspace_doc_by_location(workspace, location)
        if doc is None:
            response = self._request(
                "POST",
                f"/workspace/{self.workspace_slug}/update-embeddings",
                json={"adds": [location], "deletes": []},
            )
            body = response.json()
            if "workspace" not in body:
                raise AnythingLLMError(f"Unexpected update-embeddings response: {body}")
            workspace = self.workspace()
            doc = workspace_doc_by_location(workspace, location)
        if doc is None:
            raise AnythingLLMError(f"Workspace did not link document after update-embeddings: {location}")
        return str(doc.get("docId") or doc.get("id") or "")

    def workspace(self) -> dict:
        payload = self._request("GET", f"/workspace/{self.workspace_slug}").json()
        workspaces = payload.get("workspace")
        if not workspaces:
            raise AnythingLLMError(f"Workspace {self.workspace_slug!r} not found")
        return workspaces[0]

    def vector_search(self, query: str, top_n: int = 5, score_threshold: float = 0.0) -> dict:
        return self._request(
            "POST",
            f"/workspace/{self.workspace_slug}/vector-search",
            json={"query": query, "topN": top_n, "scoreThreshold": score_threshold},
        ).json()

    def chat(self, message: str, session_id: str) -> dict:
        return self._request(
            "POST",
            f"/workspace/{self.workspace_slug}/chat",
            json={"message": message, "mode": "query", "sessionId": session_id},
        ).json()

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        if self._failed:
            raise AnythingLLMError("AnythingLLM calls stopped after an earlier service failure")
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.request(method, path, **kwargs)
                if response.status_code in {401, 403, 404, 422}:
                    response.raise_for_status()
                if response.status_code >= 500:
                    response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                retry_after = None
                response = getattr(exc, "response", None)
                if response is not None:
                    retry_after = response.headers.get("Retry-After")
                if attempt >= self.max_attempts or _non_retryable(exc):
                    self._failed = True
                    raise AnythingLLMError(f"AnythingLLM request failed: {method} {path}: {exc}") from exc
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else min(2 * attempt, 5))
        self._failed = True
        raise AnythingLLMError(f"AnythingLLM request failed: {method} {path}: {last_exc}")

    def _stream(self, method: str, path: str, **kwargs):
        if self._failed:
            raise AnythingLLMError("AnythingLLM calls stopped after an earlier service failure")
        return self._client.stream(method, path, **kwargs)


def flatten_documents(node: dict, folder: str = "") -> list[dict]:
    if node.get("type") == "file":
        location = f"{folder}/{node['name']}".strip("/")
        return [{**node, "location": location}]
    next_folder = folder
    if node.get("name") and node.get("name") != "documents":
        next_folder = f"{folder}/{node['name']}".strip("/")
    items: list[dict] = []
    for child in node.get("items", []):
        items.extend(flatten_documents(child, next_folder))
    return items


def workspace_doc_by_location(workspace: dict, location: str) -> dict | None:
    for doc in workspace.get("documents", []):
        if doc.get("docpath") == location:
            return doc
    return None


def _non_retryable(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return response is not None and response.status_code in {400, 401, 403, 404, 422}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
