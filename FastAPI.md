# Best Practices

1. Do not use blocking operations such as I/O, Database, sleep, ... operations inside async function. Use normal function since fastAPI will run it in multi threads. The same rule for FastAPI dependencies. 

2. Use Async friendly code for example: 
    - async.sleep() instead of time.sleep()
    - async with httpx.AsyncClient() as client: instead of request.get(url)
    - AsyncIOMotorClient() instead of MongoClient() 

3. Use background tasks to not keep user waiting for long time. Do not put heavy operations in background tasks. 
4. BackGroundTasks are not guranteed to be executed. So do not use it for critical operations. No retry. fail if app crash. For those tasks use queue and worker( celery, redis queue, rabbitmq, ...).
5. Do not expose swagger docs in production. Set docs_url=None, redoc_url=None, openapi_url=None in FastAPI() constructor for pruduction environment.


6. Do not manually construct response model in the endpoints. return a plan dict and let fastAPI handle it. 

7. Do not validate data in the endpoints. Use Pydantic models for validation. You can add custom validation in the Pydantic models.

8. Use dependencies for DB-based validation. It is more efficient since fastAPI caches dependencies per request. 

9. Use connection pool for datavese connections and dependency injection to get the connection from the pool. Do not create a new connection for each request.

    For example:
    ```python

    async def lifespan(app: FastAPI):
        # create connection pool
        pool = await create_pool()
        # inject the pool into the app state
        app.state.pool = pool
        yield
        # close the pool when the app shutdown
        await pool.state.pool.close()

    async def get_connn(request: Request):
        # get the connection from the pool
        async with request.app.state.pool.acquire() as conn:
            yield conn
    
    @app.get("/items/{item_id}")
    async def read_item(item_id: int, conn=Depends(get_connn)): 

    ```

10. use the new lifespan event handler:
    ```python
    async def lifespan(app: FastAPI):
        ... # DB, Redis, ...
        yield   # app is running here
        # clore all resources here
    app = FastAPI(lifespan=lifespan)
    ```
11. use basesetting for validation and getting all the env configurations.
    ```python
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        db_url: str
        redis_url: str
        ...
    settings = Settings()
    ```
    Another option is to use dynaconf:
    ```python
    from dynaconf import Dynaconf
    settings = Dynaconf(
        settings_files=["settings.yaml"],
        environment=True,  # read from env variables
        env= 'development'  # read from settings.development.yaml
    )
    ```
    settings.yaml:
    ```yaml
    development:
        DEBUG: True
        db_url: "mongodb://localhost:27017"
        redis_url: "redis://localhost:6379"
    production:
        DEBUG: False
        db_url: "mongodb://prod-db:27017"
        redis_url: "redis://prod-redis:6379"
    ```

12. use logging instead of print statements. Use a logging library such as loguru, structlog, ... for better logging experience. Do not log sensitive information such as passwords, tokens, ... Use elasticsearch if you have a lot of logs and want to search them easily.

13. install uvloop for better performance. 

14. In production, we use gunicorn with uvicorn workers. Do not use uvicorn directly in production since it is not designed for production use.
    ```bash
    
    gunicorn main:app
    -worker 4 (cput cores * 2 + 1 )
    -worker-class uvicorn.workers.UvicornWorker 
    -bind 0.0.0:8000
    ```

# Authentication 

## Methods:

## Simple Authentication Methods:
    - Basic Authentication: username and password in the header. 
    - Digest Authentication: username and password in the header with a hash.
    - API Key Authentication: API key in the header or query parameters.
    - Session based Authentication: session ID in the cookie. The sever must store the session data in a database or in memory so it is stateful.

## Token Based Authentication Methods:
    - JWT Authentication: JSON Web Token in the header. The server does not need to store any session data since the token contains all the information. It is stateless.
        + This is inside the header -> Authorization: Bearer <token>
    - Aceess and Refresh Token Authentication: Access token in the header and refresh token in the cookie. The server needs to store the refresh token in a database or in memory so it is stateful. The access token is short lived and the refresh token is long lived. The client can use the refresh token to get a new access token when the access token expires. Refresh token is stored in HttpOnly cookie to prevent XSS attacks. Not in Local Storage.

## OAuth2 & OpenID Connect:
    - OAuth2: ***Authorization framework*** ( Not Authentication Method) that allows third-party applications to access resources on behalf of the user. It is used for authorization and not authentication. It is stateless.
    - OpenID Connect: Authentication layer on top of OAuth2. It is used for authentication and not authorization. It is stateless.

## SSO (Single Sign-On): It is a user experience not Auth method. 

    - SSO: allows users to authenticate once and access multiple applications without re-authenticating. It can be implemented using OAuth2, OpenID Connect, SAML, ... It is stateless.

# Key Concepts:

- OAuth2 defines how clients obtain an access token, and the client then sends that token to your API in the Authorization header using the Bearer scheme. A JWT is a common format for that access token. HTTP: Authorization: Bearer <access_token>

- ***OAuth2:*** an authorization framework (flows, token issuance, scopes, refresh tokens, etc.).
- ***Bearer token:*** how the access token is presented to the server (in the Authorization header).
- ***JWT:*** a token format (three base64url parts: header.payload.signature) often used as the access token.


