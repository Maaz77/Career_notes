# Reverse Proxy vs Load Balancer vs API Gateway
### Reverse Proxy
- A server that sits **in front of one or more backend servers**.
- Clients send requests to the reverse proxy; the reverse proxy **forwards** them to internal upstream servers and returns the response.
- Clients typically **do not know** the internal server addresses/ports.

### Load Balancer
- A component that **distributes traffic across multiple instances** of a service to improve availability and performance.
- Can operate at:
  - **L4 (TCP/UDP)**: forwards connections (not HTTP-aware).
  - **L7 (HTTP)**: forwards HTTP requests (HTTP-aware; can behave like a reverse proxy).

### API Gateway
- A **specialized reverse proxy for APIs** in multi-service environments.
- Provides a single entry point for clients and enforces **API-level policies** and routing across multiple backend services.

---

## Intuitive mental models
- **Reverse proxy**: “Reception desk in front of servers.”
- **Load balancer**: “Traffic dispatcher choosing which instance handles the request.”
- **API gateway**: “Front door for APIs with security/policy controls.”

---

## What a reverse proxy typically does
- Provides **one stable public endpoint** (domain stays constant while upstreams change).
- Handles **TLS/HTTPS termination** (certificates managed centrally).
- Routes requests by **host/path**:
  - `/` → frontend
  - `/api/*` → backend
  - `/admin/*` → admin service
- **Hides/protects** internal servers (private network, not directly exposed).
- Adds HTTP features (common examples):
  - caching, compression
  - header manipulation (e.g., `X-Forwarded-For`)
  - request limits, basic rate limiting (depending on setup)

---

## What the reverse proxy is NOT
- Not the application/business logic.
- It can do load balancing, but its main identity is “HTTP front door that forwards to upstreams.”

---

## Relationship between the terms
### Is a load balancer a reverse proxy?
- **L7 load balancer**: yes, it is effectively a reverse proxy (HTTP-aware request forwarding).
- **L4 load balancer**: usually not called a reverse proxy (it forwards connections, not HTTP requests).

### Is an API gateway a reverse proxy?
- Yes. It is a reverse proxy **plus** API-specific features, commonly:
  - authentication integration (JWT/OIDC validation)
  - authorization checks (scopes/claims), quotas/rate limiting
  - API versioning and routing across many services
  - centralized observability (logging/metrics/tracing)
  - optional transformations/standardized error handling

---

## One-sentence test for “reverse proxy”
If a component receives the public request, chooses an internal upstream, forwards the request, and returns the response, it is acting as a **reverse proxy**.



# AWS

- ***IAM Role***: Similar to user (an identity with permission), does not have credentials (key, pass), assumebly, temprorarily, by anyone who needs it. 
    - examples of roles: Read from s3 write to cloudwatch, read from dynamodb, write to dynamodb, ...
    - roles can be assigned to EC2 instances, Lambda functions, ECS tasks, ... and the users.
- ***IAM Policy***: Who cad do what to which resource and when.
    - example: Allow IAM users to rotate their own credentials programatically and in the console.
    - example: Allow a Lambda function to access a dynamodb table.
    - example: Allow a user to start and stop EC2 instances.
    - we attach policies to users, groups, and roles. 

    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "iam:ChangePassword", 
                "Resource": "arn:aws:iam::123456789012:user/${aws:username}"
            }
        ]
    }
    ```
- ***arn***: Amazon Resource Name, a unique identifier for AWS resources. It has the format: arn:partition:service:region:account-id:resource-type/resource-id
    - example: arn:aws:s3:::my-bucket/my-object
    - example: arn:aws:iam::123456789012:user/JohnDoe

- ***Lambda***: The lambda function gets two parameters: event and context. The event contains the data passed to the function, and the context contains information about the execution environment.