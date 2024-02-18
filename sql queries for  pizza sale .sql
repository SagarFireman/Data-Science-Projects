create database pizza_sales;
use pizza_sales;
select * from pizza_sales;

select sum(total_price) as total_revenue from pizza_sales;

select sum(total_price)/count(distinct order_id) as avg_order_value from pizza_sales;

select sum(quantity) as toptal_pizza_sold from pizza_sales;

select count(distinct order_id) as total_orders from pizza_sales;

select sum(quantity)/ count(distinct order_id) from pizza_sales;

SELECT 
    pizza_name,
    SUM(quantity) AS total_quantity
FROM pizza_sales
GROUP BY pizza_name
ORDER BY total_quantity DESC
LIMIT 5;

SELECT 
    pizza_size,
    AVG(unit_price) AS average_unit_price
FROM pizza_sales
GROUP BY pizza_size;

SELECT AVG(quantity) AS average_quantity_per_order
FROM pizza_sales;

SELECT 
    pizza_name,
    SUM(total_price) AS total_revenue
FROM 
    pizza_sales
GROUP BY 
    pizza_name;
    

SELECT SUBSTRING(order_time, 1, 2) AS hour_of_day,
       COUNT(order_id) AS orders_count
FROM 
    pizza_sales
GROUP BY 
    hour_of_day
ORDER BY 
    orders_count DESC;
    
SELECT 
    pizza_category,
    SUM(total_price) AS total_sales,
    (SUM(total_price) / (SELECT SUM(total_price) FROM pizza_sales)) * 100 AS sales_percentage
FROM 
    pizza_sales
GROUP BY 
    pizza_category
ORDER BY 
    total_sales DESC;
