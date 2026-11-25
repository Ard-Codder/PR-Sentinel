"""Helpers for checkout totals."""


def customer_name(customer):
    name = customer.get("name") if customer else None
    return name.strip()


def unit_price(line):
    price = line.get("price")
    quantity = line.get("quantity")
    return price / quantity


def average_price(lines):
    total = 0
    for line in lines:
        total += line["price"]
    return total / len(lines)


def discounted_total(total, percent):
    try:
        return total - (total * int(percent) / 100)
    except:
        return 0
