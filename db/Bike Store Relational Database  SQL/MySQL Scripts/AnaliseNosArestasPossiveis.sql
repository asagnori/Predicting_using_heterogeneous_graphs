USE bike_store;

-- 1 Contar nós por tabela
SELECT COUNT(*) FROM orders;
SELECT COUNT(*) FROM order_items;
SELECT COUNT(*) FROM customers;
SELECT COUNT(*) FROM products;
SELECT COUNT(*) FROM stores;
SELECT COUNT(*) FROM staffs;

-- 2 Contar arestas (relações) 

-- order -> customer
SELECT COUNT(*) 
FROM orders 
WHERE customer_id IS NOT NULL;

-- order_items -> orders
SELECT COUNT(*) 
FROM order_items 
WHERE order_id IS NOT NULL;

-- order_items -> products
SELECT COUNT(*) 
FROM order_items 
WHERE product_id IS NOT NULL;

-- order -> stores
SELECT COUNT(*) 
FROM orders 
WHERE store_id IS NOT NULL; 

-- order -> staffs
SELECT COUNT(*) 
FROM orders 
WHERE staff_id IS NOT NULL;

-- product > categories
SELECT COUNT(*) 
FROM products
WHERE category_id IS NOT NULL;

-- product > brands
SELECT COUNT(*) 
FROM products
WHERE brand_id IS NOT NULL;

-- Contando o Total de Arestas (Conexões no Grafo)
SELECT 
    (SELECT COUNT(*) FROM orders WHERE customer_id IS NOT NULL) +  
    (SELECT COUNT(*) FROM orders WHERE store_id IS NOT NULL) +     
    (SELECT COUNT(*) FROM orders WHERE staff_id IS NOT NULL) +     
    (SELECT COUNT(*) FROM order_items WHERE order_id IS NOT NULL) +    
    (SELECT COUNT(*) FROM order_items WHERE product_id IS NOT NULL) +  
    (SELECT COUNT(*) FROM staffs WHERE store_id IS NOT NULL) + 
    (SELECT COUNT(*) FROM products WHERE brand_id IS NOT NULL) +
    (SELECT COUNT(*) FROM products WHERE category_id IS NOT NULL) +
    (SELECT COUNT(*) FROM stocks)
AS Total_Soma_Arestas_E;
