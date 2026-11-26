# FastAPI

Source: Wikipedia, article 'FastAPI' (public, CC BY-SA)

FastAPI is a web framework for building HTTP-based service APIs in Python 3.8+.  It uses Pydantic and type hints to validate, serialize and deserialize data. FastAPI also automatically generates OpenAPI documentation for APIs built with it. It was first released in 2018.


== Components ==


=== Pydantic ===
Pydantic is a data validation library for Python. While writing code in an IDE, Pydantic provides type hints based on annotations. FastAPI extensively utilizes Pydantic models for data validation, serialization, and automatic API documentation. These models are using standard Python type hints, providing a declarative way to specify the structure and types of data for incoming requests (e.g., HTTP bodies) and outgoing responses.


=== Starlette ===
Starlette is a lightweight ASGI framework/toolkit, to support async functionality in Python.


=== Uvicorn ===
Uvicorn is a minimal low-level server/application web server for async frameworks, following the ASGI specification. Technically, it implements a multi-process model with one main process, which is responsible for managing a pool of worker processes and distributing incoming HTTP requests to them. The number of worker processes is pre-configured, but can also be adjusted up or down at runtime.


=== OpenAPI integration ===
FastAPI automatically generates OpenAPI documentation for APIs. This documentation includes both Swagger UI and ReDoc, which provide interactive API documentation that you can use to explore and test your endpoints in real time. This is particularly useful for developing, testing, and sharing APIs with other developers or users. Swagger UI is accessible by default at /docs and ReDoc at /redoc route.


== Features ==


=== Asynchronous operations ===
FastAPI's architecture inherently supports asynchronous programming. This design allows the single-threaded event loop to handle a large number of concurrent requests efficiently, particularly when dealing with I/O-bound operations like database queries or external API calls. For reference, see async/await pattern.


=== Dependency injection ===
FastAPI incorporates a Dependency Injection (DI) system to manage and provide services to HTTP endpoints. This mechanism allows developers to declare components such as database sessions or authentication logic as function parameters. FastAPI automatically resolves these dependencies for each request, injecting the necessary instances.


=== WebSockets support ===
WebSockets allow full-duplex communication between a client and the server. This capability is fundamental for applications requiring continuous data exchange, such as instant messaging platforms, live data dashboards, or multiplayer online games. FastAPI leverages the underlying Starlette implementation, allowing for efficient management of connections and message handling.


=== Background tasks ===
FastAPI enables the execution of background tasks after an HTTP response has been sent to the client. This allows the API to immediately respond to user requests while simultaneously processing non-critical or time-consuming operations in the background. Typical applications include sending email notifications, updating caches, or performing data post-processing.


== Example ==
The following code shows a simple web application that displays "Hello, World!" when visited:


== See also ==

Django (web framework)
Flask (web framework)
Pylons project
Web2py
Tornado (web server)
Comparison of server-side web frameworks § Python
REST
Python (programming language)


== References ==


== External links ==
Official website 
fastapi on GitHub