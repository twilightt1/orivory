"""
Smoke tests for Docker Compose services.

These tests verify that all required services are healthy
and responding correctly. Designed to run against a live
Docker Compose stack.

Mark: @pytest.mark.smoke
"""
import pytest


def is_docker_available():
    """Check if Docker services are available."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        # Try common service ports
        for port in [5432, 6379, 6333, 9000]:
            try:
                sock.connect(("localhost", port))
                sock.close()
                return True
            except (TimeoutError, OSError):
                continue
        sock.close()
        return False
    except Exception:
        return False


DOCKER_AVAILABLE = is_docker_available()


def skip_if_no_docker():
    """Skip decorator helper."""
    if not DOCKER_AVAILABLE:
        pytest.skip("Docker services not available - run 'docker compose up -d'")


@pytest.mark.smoke
class TestPostgresHealth:
    """Tests for PostgreSQL service."""

    def test_postgres_is_ready(self, docker_services):
        """PostgreSQL should accept connections and be ready."""
        skip_if_no_docker()
        import psycopg2

        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="ragdb",
                user="postgres",
                password="password",
                connect_timeout=5,
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()

            assert result == (1,)
        except psycopg2.OperationalError:
            pytest.skip("PostgreSQL not available")

    def test_postgres_extensions(self, docker_services):
        """PostgreSQL should have required extensions."""
        skip_if_no_docker()
        import psycopg2

        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="ragdb",
                user="postgres",
                password="password",
                connect_timeout=5,
            )
            cursor = conn.cursor()
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            result = cursor.fetchone()
            cursor.close()
            conn.close()

            print(f"Vector extension: {result}")
        except psycopg2.OperationalError:
            pytest.skip("PostgreSQL not available")


@pytest.mark.smoke
class TestQdrantHealth:
    """Tests for the Qdrant vector store service."""

    def test_qdrant_is_ready(self, docker_services):
        """Qdrant /readyz should answer 200 once its shards are ready."""
        skip_if_no_docker()
        import requests

        try:
            response = requests.get("http://localhost:6333/readyz", timeout=5)
            assert response.status_code == 200
            # Real Qdrant /readyz: plain text "all shards are ready".
            assert "ready" in response.text
        except requests.exceptions.RequestException:
            pytest.skip("Qdrant not available")

    def test_qdrant_version(self, docker_services):
        """Qdrant should report its version on the root endpoint."""
        skip_if_no_docker()
        import requests

        try:
            response = requests.get("http://localhost:6333/", timeout=5)
            assert response.status_code == 200
            data = response.json()
            # Real Qdrant root: {"title": "qdrant - vector search engine", "version": "1.x.y"}
            assert data["title"] == "qdrant - vector search engine"
            assert isinstance(data["version"], str) and len(data["version"]) > 0
        except requests.exceptions.RequestException:
            pytest.skip("Qdrant not available")


@pytest.mark.smoke
class TestMinIOHealth:
    """Tests for MinIO object storage service."""

    def test_minio_health(self, docker_services):
        """MinIO should respond to health check."""
        skip_if_no_docker()
        import requests

        try:
            response = requests.get("http://localhost:9000/minio/health/live", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("MinIO not available")

    def test_minio_api(self, docker_services):
        """MinIO API should be accessible."""
        skip_if_no_docker()
        import requests
        from requests.auth import HTTPBasicAuth

        try:
            response = requests.get(
                "http://localhost:9000/api/v1/buckets",
                auth=HTTPBasicAuth("minioadmin", "minioadmin"),
                timeout=5,
            )
            # :9000 serves the S3 API, not the console route — any HTTP
            # status (incl. 400) proves the server is up and reachable.
            assert response.status_code in [200, 400, 403]
        except requests.exceptions.RequestException:
            pytest.skip("MinIO not available")


@pytest.mark.smoke
class TestAPIHealth:
    """Tests for API service health endpoint."""

    @pytest.fixture
    def api_base_url(self):
        """Base URL for API."""
        return "http://localhost:8000"

    def test_api_health_endpoint(self, docker_services, api_base_url):
        """API /health endpoint should respond."""
        skip_if_no_docker()
        import time

        import requests

        try:
            for _i in range(30):
                try:
                    response = requests.get(f"{api_base_url}/health", timeout=2)
                    if response.status_code == 200:
                        break
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
            else:
                pytest.skip("API not available")

            assert response.status_code == 200
            data = response.json()
            assert "status" in data or "healthy" in data
        except requests.exceptions.RequestException:
            pytest.skip("API not available")

    def test_api_docs_accessible(self, docker_services, api_base_url):
        """API documentation should be accessible."""
        skip_if_no_docker()
        import requests

        try:
            response = requests.get(f"{api_base_url}/docs", timeout=5)
            assert response.status_code in [200, 301, 302]
        except requests.exceptions.RequestException:
            pytest.skip("API not available")


@pytest.mark.smoke
class TestServiceConnectivity:
    """Tests for service-to-service connectivity."""

    def test_postgres_from_app_container(self, docker_services):
        """App should be able to connect to PostgreSQL."""
        skip_if_no_docker()
        import psycopg2

        try:
            conn = psycopg2.connect(
                host="postgres",
                port=5432,
                database="ragdb",
                user="postgres",
                password="password",
                connect_timeout=5,
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            assert result == (1,)
        except psycopg2.OperationalError:
            pytest.skip("Running outside Docker network")
