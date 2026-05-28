def load_data(engine):
    """Load data from the database."""
    with engine.connect() as connection:
        result = connection.execute("SELECT * FROM your_table")
        data = result.fetchall()
    return data