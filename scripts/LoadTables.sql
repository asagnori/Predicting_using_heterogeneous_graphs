--- Load Data into Tables ---

use bike_store;
SET GLOBAL local_infile = 1;

SHOW VARIABLES LIKE 'pid_file';
SHOW VARIABLES LIKE 'local_infile';
 
--- Categories ---
LOAD DATA LOCAL INFILE '/Users/angelosagnori/Downloads/0- MBA USP-ESALQ/TCC MBA USP-ESALQ/Projeto Pesquisa/datasets/Bike Store Relational Database  SQL/main-db_archive/categories.csv'
INTO TABLE categories
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(category_id, category_name);

--- brands ---
LOAD DATA LOCAL INFILE '/Users/angelosagnori/Downloads/0- MBA USP-ESALQ/TCC MBA USP-ESALQ/Projeto Pesquisa/datasets/Bike Store Relational Database  SQL/main-db_archive/brands.csv'
INTO TABLE brands
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(brand_id, brand_name);

--- products ---
LOAD DATA LOCAL INFILE '/Users/angelosagnori/Downloads/0- MBA USP-ESALQ/TCC MBA USP-ESALQ/Projeto Pesquisa/datasets/Bike Store Relational Database  SQL/main-db_archive/products.csv'
INTO TABLE products
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(product_id,
 product_name,
 brand_id,
 category_id,
 model_year,
 list_price);

--- customers ---

LOAD DATA LOCAL INFILE '/Users/angelosagnori/Downloads/0- MBA USP-ESALQ/TCC MBA USP-ESALQ/Projeto Pesquisa/datasets/Bike Store Relational Database  SQL/main-db_archive/customers.csv'
INTO TABLE customers
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(customer_id,
 first_name,
 last_name,
 phone,
 email,
 street,
 city,
 state,
 zip_code);

--- stores ---

LOAD DATA LOCAL INFILE '/Users/angelosagnori/Downloads/0- MBA USP-ESALQ/TCC MBA USP-ESALQ/Projeto Pesquisa/datasets/Bike Store Relational Database  SQL/main-db_archive/stores.csv'
INTO TABLE stores
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(store_id,
 store_name,
 phone,
 email,
 street,
 city,
 state,
 zip_code);

--- staffs --- 

SET FOREIGN_KEY_CHECKS = 0;

LOAD DATA LOCAL INFILE '/Users/angelosagnori/Downloads/0- MBA USP-ESALQ/TCC MBA USP-ESALQ/Projeto Pesquisa/datasets/Bike Store Relational Database  SQL/main-db_archive/staffs.csv'

INTO TABLE staffs
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(
 staff_id,
 first_name,
 last_name,
 email,
 phone,
 active,
 store_id,
 @manager_id
)
SET manager_id =
    CASE
        WHEN TRIM(@manager_id) REGEXP '^[0-9]+$'
        THEN CAST(TRIM(@manager_id) AS UNSIGNED)
        ELSE NULL
    END;

SET FOREIGN_KEY_CHECKS = 1;

--- orders ---
LOAD DATA LOCAL INFILE '/Users/angelosagnori/Downloads/0- MBA USP-ESALQ/TCC MBA USP-ESALQ/Projeto Pesquisa/datasets/Bike Store Relational Database  SQL/main-db_archive/orders.csv'
INTO TABLE orders
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(order_id,
 customer_id,
 order_status,
 order_date,
 required_date,
 shipped_date,
 store_id,
 staff_id);
 
--- orders items ---
LOAD DATA LOCAL INFILE '/Users/angelosagnori/Downloads/0- MBA USP-ESALQ/TCC MBA USP-ESALQ/Projeto Pesquisa/datasets/Bike Store Relational Database  SQL/main-db_archive/order_items.csv'
INTO TABLE order_items
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(order_id,
 item_id,
 product_id,
 quantity,
 list_price,
 discount);

--- stocks ---

LOAD DATA LOCAL INFILE '/Users/angelosagnori/Downloads/0- MBA USP-ESALQ/TCC MBA USP-ESALQ/Projeto Pesquisa/datasets/Bike Store Relational Database  SQL/main-db_archive/stocks.csv'
INTO TABLE stocks
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(store_id,
 product_id,
 quantity);

 



