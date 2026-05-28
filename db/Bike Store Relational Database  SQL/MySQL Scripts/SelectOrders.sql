Use bike_store;

SELECT * FROM bike_store.orders;
SELECT * FROM bike_store.order_items;

-- Total de Ordens e Items ---
SELECT
    COUNT(DISTINCT o.order_id)       AS total_orders,
    COUNT(*)                         AS total_rows_join
FROM orders o
JOIN order_items oi
    ON o.order_id = oi.order_id;

--- Percentual de Atraso ---    
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN shipped_date > required_date THEN 1 ELSE 0 END) as atrasados,
    (SUM(CASE WHEN shipped_date > required_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as pct_atraso
FROM orders
WHERE shipped_date IS NOT NULL;

