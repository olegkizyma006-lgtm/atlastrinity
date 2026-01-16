#!/usr/bin/env python3

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Premium Seafood E-Commerce API",
    description="API for premium seafood e-commerce platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from catalog.routes import products, categories
from cart.routes import cart, cart_items
from payments.routes import payments, webhooks
from logistics.routes import shipping, tracking

# Include all routers
app.include_router(products.router, prefix="/api/products", tags=["Products"])
app.include_router(categories.router, prefix="/api/categories", tags=["Categories"])
app.include_router(cart.router, prefix="/api/cart", tags=["Cart"])
app.include_router(cart_items.router, prefix="/api/cart/items", tags=["Cart Items"])
app.include_router(payments.router, prefix="/api/payments", tags=["Payments"])
app.include_router(webhooks.router, prefix="/api/webhooks", tags=["Webhooks"])
app.include_router(shipping.router, prefix="/api/shipping", tags=["Shipping"])
app.include_router(tracking.router, prefix="/api/tracking", tags=["Tracking"])


@app.get("/")
def read_root():
    return {
        "message": "Welcome to Premium Seafood E-Commerce API",
        "version": "1.0.0",
        "services": [
            "catalog",
            "cart", 
            "payments",
            "logistics"
        ]
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": "connected",  # This should be checked dynamically
        "services": {
            "catalog": "operational",
            "cart": "operational",
            "payments": "operational",
            "logistics": "operational"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info")
    )