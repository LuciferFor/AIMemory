from fastapi.testclient import TestClient

from aimemory.core.config import get_settings
from aimemory.db.session import get_db
from aimemory.main import create_app


class _Rows:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list:
        return self._rows


class _FakeDb:
    def scalar(self, query) -> int:
        return 0

    def execute(self, query) -> _Rows:
        return _Rows([("ready", 0), ("pending", 0), ("failed", 0)])

    def scalars(self, query) -> _Rows:
        return _Rows([])


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-secret")
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: _FakeDb()
    return TestClient(app)


def test_admin_requires_login(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.get("/admin", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/login")


def test_root_redirects_to_admin_login(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/admin/login"


def test_root_head_redirects_to_admin_login(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.head("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/admin/login"


def test_admin_login_success(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.post(
        "/admin/login",
        data={"username": "admin", "password": "secret", "next": "/admin"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/admin"
    assert "aimemory_admin" in response.headers["set-cookie"]


def test_admin_login_failure(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.post(
        "/admin/login",
        data={"username": "admin", "password": "wrong", "next": "/admin"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/login")


def test_admin_dashboard_for_logged_in_user(monkeypatch) -> None:
    client = _client(monkeypatch)
    client.post("/admin/login", data={"username": "admin", "password": "secret", "next": "/admin"})

    response = client.get("/admin")

    assert response.status_code == 200
    assert "仪表盘" in response.text
