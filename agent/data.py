import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# Load enviornment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

sessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(primary_key=True)
    customer_id: Mapped[str]
    status: Mapped[str]
    amount: Mapped[float]
    refund_eligible: Mapped[bool]
    days_since_purchase: Mapped[int]


def seed_if_empty():
    # Create the table if it doesn't exist yet. Safe to call every time —
    # create_all() is a no-op for tables that already exist.
    Base.metadata.create_all(engine)

    with sessionLocal() as session:
        # Don't reseed on every run — if there's already a row, bail out.
        if session.query(Order).first() is not None:
            return

        session.add_all(
            [
                # Normal happy-path orders, well within the 15-day window.
                Order(
                    order_id="A1001",
                    customer_id="C001",
                    status="delivered",
                    amount=49.99,
                    refund_eligible=True,
                    days_since_purchase=3,
                ),
                Order(
                    order_id="A1002",
                    customer_id="C002",
                    status="delivered",
                    amount=89.50,
                    refund_eligible=True,
                    days_since_purchase=10,
                ),
                # Boundary: exactly at the 15-day cutoff -> still eligible.
                Order(
                    order_id="A1003",
                    customer_id="C003",
                    status="delivered",
                    amount=25.00,
                    refund_eligible=True,
                    days_since_purchase=15,
                ),
                # Boundary: one day past the cutoff -> no longer eligible.
                Order(
                    order_id="A1004",
                    customer_id="C004",
                    status="delivered",
                    amount=60.00,
                    refund_eligible=False,
                    days_since_purchase=16,
                ),
                # Negative case: already refunded, status blocks it regardless of days.
                Order(
                    order_id="A1005",
                    customer_id="C005",
                    status="refunded",
                    amount=40.00,
                    refund_eligible=False,
                    days_since_purchase=5,
                ),
                # Negative case: cancelled order.
                Order(
                    order_id="A1006",
                    customer_id="C006",
                    status="cancelled",
                    amount=75.00,
                    refund_eligible=False,
                    days_since_purchase=2,
                ),
                # Edge case: not delivered yet, refund doesn't apply.
                Order(
                    order_id="A1007",
                    customer_id="C007",
                    status="in_transit",
                    amount=120.00,
                    refund_eligible=False,
                    days_since_purchase=1,
                ),
                # High-amount order, useful later for a guardrail check.
                Order(
                    order_id="A1008",
                    customer_id="C008",
                    status="delivered",
                    amount=999.99,
                    refund_eligible=True,
                    days_since_purchase=1,
                ),
            ]
        )
        session.commit()


def fetch_order(order_id: str) -> dict | None:
    with sessionLocal() as session:
        order = session.get(Order, order_id)
        if order is None:
            return None
        return {
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "status": order.status,
            "amount": order.amount,
            "refund_eligible": order.refund_eligible,
            "days_since_purchase": order.days_since_purchase,
        }


if __name__ == "__main__":
    seed_if_empty()
    print(fetch_order("A1003"))  # existing row -> dict
    print(fetch_order("nope-999"))  # missing row -> None
