# Docker Setup for Storage Tests

This document contains Docker commands and configuration for running storage provider tests.

## Quick Start

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Stop all services:**
   ```bash
   docker-compose down
   ```

3. **Remove volumes (clean data):**
   ```bash
   docker-compose down -v
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

## Environment Variables

Create a `.env` file in the project root with:

```bash
# PostgreSQL Configuration
POSTGRES_USER=upsonic_test
POSTGRES_PASSWORD=test_password
POSTGRES_DB=upsonic_test
POSTGRES_PORT=5432
POSTGRES_URL=postgresql://upsonic_test:test_password@localhost:5432/upsonic_test

# MongoDB Configuration
MONGO_USER=upsonic_test
MONGO_PASSWORD=test_password
MONGO_PORT=27017
MONGO_URL=mongodb://upsonic_test:test_password@localhost:27017/?authSource=admin

# Redis Configuration
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379/0
```

## Individual Service Commands

### PostgreSQL

**Start:**
```bash
docker run -d \
  --name upsonic_test_postgres \
  -e POSTGRES_USER=upsonic_test \
  -e POSTGRES_PASSWORD=test_password \
  -e POSTGRES_DB=upsonic_test \
  -p 5432:5432 \
  postgres:15-alpine
```

**Stop:**
```bash
docker stop upsonic_test_postgres
docker rm upsonic_test_postgres
```

### MongoDB

**Start:**
```bash
docker run -d \
  --name upsonic_test_mongo \
  -e MONGO_INITDB_ROOT_USERNAME=upsonic_test \
  -e MONGO_INITDB_ROOT_PASSWORD=test_password \
  -p 27017:27017 \
  mongo:7
```

**Stop:**
```bash
docker stop upsonic_test_mongo
docker rm upsonic_test_mongo
```

### Redis

**Start:**
```bash
docker run -d \
  --name upsonic_test_redis \
  -p 6379:6379 \
  redis:7-alpine
```

**Stop:**
```bash
docker stop upsonic_test_redis
docker rm upsonic_test_redis
```

## Health Checks

Check if services are running:

```bash
# PostgreSQL
docker exec upsonic_test_postgres pg_isready -U upsonic_test

# MongoDB
docker exec upsonic_test_mongo mongosh --eval "db.adminCommand('ping')"

# Redis
docker exec upsonic_test_redis redis-cli ping
```

## Notes

- Services use test credentials and are intended for development/testing only
- Data persists in Docker volumes unless explicitly removed
- Ports can be customized via environment variables

