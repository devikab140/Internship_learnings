DROP DATABASE Bank;
CREATE DATABASE Bank;
USE Bank;
CREATE TABLE customer (
  customer_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  first_name VARCHAR(100) NOT NULL,
  last_name VARCHAR(100) NOT NULL,
  dob DATE NOT NULL,
  gender VARCHAR(20),
  phone VARCHAR(20) UNIQUE,
  email VARCHAR(150) UNIQUE,
  address TEXT,
  proof_type VARCHAR(50),
  proof_number VARCHAR(100),
  nominee_customer_id INT UNSIGNED NULL,  -- internal nominee (references another customer)
  nominee_name VARCHAR(150),              -- external nominee name (if not internal)
  nominee_relation VARCHAR(50),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  CONSTRAINT fk_customer_nominee
    FOREIGN KEY (nominee_customer_id) REFERENCES customer(customer_id)
);


INSERT INTO customer 
(first_name, last_name, dob, gender, phone, email, address, proof_type, proof_number,nominee_customer_id, nominee_name, nominee_relation)
VALUES
('Sree', 'Kumar', '1995-08-12', 'Female', '+91-9876543210', 'deva@example.com', '12 MG Road, Bangalore', 'Aadhar', '1234-5678-9012', NULL, 'Rahul Sharma','Brother'),
('Rahul', 'Sharma', '1988-05-20', 'Male', '+91-9123456780', 'rahul@example.com', '45 Park Street, Delhi', 'PAN', 'ABCDE1234F', 1, NULL, NULL),
('Sangeeta', 'Rao', '1992-11-03', 'Female', '+91-9988776655', 'sangeeta@example.com', '78 MG Road, Bangalore', 'Passport', 'M1234567',2,NULL,NULL),
('Amit', 'Gupta', '1985-02-14', 'Male', '+91-9876501234', 'amit@example.com', '56 Residency Road, Bangalore', 'Aadhar', '4321-8765-2109', NULL, 'Devasree Kumar', 'Friend'),
('Priya', 'Menon', '1990-06-25', 'Female', '+91-9998887776', 'priya@example.com', '23 Park Avenue, Delhi', 'PAN', 'XYZ9876A', 3, NULL, NULL),
('Vikram', 'Singh', '1987-09-12', 'Male', '+91-9112233445', 'vikram@example.com', '12 MG Road, Chennai', 'Passport', 'P7654321', NULL, 'Amit Gupta', 'Friend'),
('Neha', 'Kapoor', '1993-03-30', 'Female', '+91-9001122334', 'neha@example.com', '90 Park Lane, Delhi', 'Aadhar', '1111-2222-3333', 5, NULL, NULL),
('Rohan', 'Kumar', '1991-07-07', 'Male', '+91-9887766554', 'rohan@example.com', '45 Residency Road, Bangalore', 'PAN', 'AB1234567C', NULL, 'Priya Menon', 'Cousin'),
('Isha', 'Verma', '1994-12-12', 'Female', '+91-9776655443', 'isha@example.com', '78 MG Road, Delhi', 'Aadhar', '5678-1234-8765', 4, NULL, NULL),
('Aditya', 'Reddy', '1989-08-18', 'Male', '+91-9665544332', 'aditya@example.com', '34 Park Street, Chennai', 'Passport', 'Q1234567', NULL, 'Neha Kapoor', 'Sister'),
('Kavita', 'Shah', '1992-01-20', 'Female', '+91-9554433221', 'kavita@example.com', '12 Residency Road, Mumbai', 'PAN', 'LMN3456K', 6, NULL, NULL),
('Saurabh', 'Joshi', '1986-04-15', 'Male', '+91-9443322110', 'saurabh@example.com', '56 MG Road, Pune', 'Aadhar', '2222-3333-4444', NULL, 'Isha Verma', 'Friend'),
('Anjali', 'Patel', '1991-11-28', 'Female', '+91-9332211009', 'anjali@example.com', '90 Park Lane, Mumbai', 'PAN', 'OPQ5678R', 7, NULL, NULL),
('Manish', 'Mehta', '1988-09-05', 'Male', '+91-9221100998', 'manish@example.com', '23 MG Road, Delhi', 'Passport', 'R7654321', NULL, 'Rohan Kumar', 'Friend'),
('Pooja', 'Sharma', '1993-06-17', 'Female', '+91-9110099887', 'pooja@example.com', '78 Park Avenue, Bangalore', 'Aadhar', '3333-4444-5555', 8, NULL, NULL);

CREATE TABLE sections (
    section_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description VARCHAR(255)
) ;


INSERT INTO sections (name, description) VALUES
('Loan', 'Handles loan processing and approvals'),
('Card', 'Manages credit and debit card services'),
('Accounts', 'Responsible for account operations and maintenance'),
('Investment', 'Deals with investment products and advisory'),
('HR', 'Human resources and employee management'),
('Security', 'Ensures branch and data security'),
('R&D', 'Research and development for new products'),
('Supply Chain', 'Manages logistics and supply operations');



CREATE TABLE branch (
    branch_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    branch_name VARCHAR(150),
    address TEXT,
    ifsc_code VARCHAR(20) UNIQUE,
    section_id INT UNSIGNED NOT NULL,
    manager_id INT UNSIGNED NULL,  -- will add FK later
    CONSTRAINT fk_branch_section FOREIGN KEY (section_id) REFERENCES sections(section_id)
);

INSERT INTO branch (branch_name, address, ifsc_code, section_id, manager_id) VALUES
('MG Road Branch', '12 MG Road, Bangalore', 'BKID0000123', 1, NULL),
('Park Street Branch', '45 Park Street, Delhi', 'BKID0000456', 2, NULL),
('Residency Road Branch', '56 Residency Road, Bangalore', 'BKID0000789', 1, NULL),
('Marathahalli Branch', '23 Marathahalli, Bangalore', 'BKID0000110', 3, NULL),
('Andheri Branch', '90 Andheri West, Mumbai', 'BKID0000221', 2, NULL),
('Koramangala Branch', '34 Koramangala, Bangalore', 'BKID0000332', 4, NULL),
('Jayanagar Branch', '78 Jayanagar, Bangalore', 'BKID0000443', 1, NULL),
('Bandra Branch', '12 Bandra West, Mumbai', 'BKID0000554', 2, NULL);

CREATE TABLE staff (
    staff_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    role VARCHAR(50),
    section_id INT UNSIGNED NOT NULL,
    branch_id INT UNSIGNED NOT NULL,
    phone VARCHAR(20) UNIQUE,
    email VARCHAR(150) UNIQUE,
    salary DECIMAL(15,2),
    join_date DATE,
    
    CONSTRAINT fk_staff_section FOREIGN KEY (section_id) REFERENCES sections(section_id),
    CONSTRAINT fk_staff_branch FOREIGN KEY (branch_id) REFERENCES branch(branch_id)
);


INSERT INTO staff (first_name, last_name, role, section_id, branch_id, phone, email, salary, join_date) VALUES
('Amit', 'Sharma', 'Loan Officer', 1, 1, '9871112222', 'amit@bank.com', 45000.00, '2020-05-01'),
('Neha', 'Kapoor', 'Card Manager', 2, 2, '9881112223', 'neha@bank.com', 50000.00, '2019-08-15'),
('Rohan', 'Kumar', 'Accounts Executive', 3, 3, '9891112224', 'rohan@bank.com', 40000.00, '2021-01-10'),
('Priya', 'Menon', 'Investment Advisor', 4, 4, '9901112225', 'priya@bank.com', 55000.00, '2018-11-20'),
('Vikram', 'Singh', 'Loan Officer', 1, 1, '9911112226', 'vikram@bank.com', 47000.00, '2020-09-05'),
('Kavita', 'Shah', 'Card Executive', 2, 2, '9921112227', 'kavita@bank.com', 42000.00, '2019-02-12'),
('Saurabh', 'Joshi', 'Accounts Manager', 3, 3, '9931112228', 'saurabh@bank.com', 60000.00, '2017-06-18'),
('Anjali', 'Patel', 'Investment Executive', 4, 4, '9941112229', 'anjali@bank.com', 48000.00, '2021-03-22'),
('Manish', 'Mehta', 'Loan Manager', 1, 1, '9951112230', 'manish@bank.com', 65000.00, '2016-12-01'),
('Isha', 'Verma', 'Card Officer', 2, 2, '9961112231', 'isha@bank.com', 46000.00, '2020-07-07'),
('Aditya', 'Reddy', 'Accounts Officer', 3, 3, '9971112232', 'aditya@bank.com', 43000.00, '2018-04-15'),
('Devasree', 'Kumar', 'Investment Manager', 4, 4, '9981112233', 'devasree@bank.com', 70000.00, '2015-10-30'),
('Rahul', 'Sharma', 'Loan Officer', 1, 1, '9991112234', 'rahul@bank.com', 47000.00, '2019-11-05'),
('Sangeeta', 'Rao', 'Card Executive', 2, 2, '9001112235', 'sangeeta@bank.com', 42000.00, '2020-01-25'),
('Amit', 'Gupta', 'Accounts Executive', 3, 3, '9011112236', 'amitg@bank.com', 44000.00, '2017-09-12');


ALTER TABLE branch
ADD CONSTRAINT fk_branch_manager 
FOREIGN KEY (manager_id) REFERENCES staff(staff_id);

UPDATE branch SET manager_id = 9 WHERE branch_id = 1;
UPDATE branch SET manager_id = 2 WHERE branch_id = 2;
UPDATE branch SET manager_id = 7 WHERE branch_id = 3;
UPDATE branch SET manager_id = 12 WHERE branch_id = 8;
UPDATE branch SET manager_id = 3 WHERE branch_id = 5;
UPDATE branch SET manager_id = 5 WHERE branch_id = 6;


-- SELECT branch_id FROM branch;

CREATE TABLE account (
    account_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    customer_id INT UNSIGNED NOT NULL,
    account_num VARCHAR(32) UNIQUE NOT NULL,
    account_type VARCHAR(50),
    balance DECIMAL(15,2) DEFAULT 0.00,
    currency CHAR(3) DEFAULT 'INR',
    interest_rate DECIMAL(5,3),
    card_issued BOOLEAN DEFAULT FALSE,
    cheque_book_issued BOOLEAN DEFAULT FALSE,
    preferred_payment_mode VARCHAR(20),
    opened_date DATE,
    branch_id INT UNSIGNED NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    
    CONSTRAINT fk_account_customer FOREIGN KEY (customer_id) REFERENCES customer(customer_id),
    CONSTRAINT fk_account_branch FOREIGN KEY (branch_id) REFERENCES branch(branch_id)
) ;


INSERT INTO account (customer_id, account_num, account_type, balance, currency, interest_rate, card_issued, cheque_book_issued, preferred_payment_mode, opened_date, branch_id, status) VALUES
(1, 'IN001000101', 'Savings', 12500.50, 'INR', 3.500, TRUE, FALSE, 'Online', '2024-02-01', 1, 'active'),
(1, 'IN001000116', 'Current', 5000.00, 'INR', 0.000, TRUE, TRUE, 'Cheque', '2024-03-01', 2, 'active'),
(4, 'IN001000102', 'Current', 50000.00, 'INR', 0.000, TRUE, TRUE, 'Cheque', '2023-11-15', 3, 'active'),
(6, 'IN001000103', 'Savings', 23000.75, 'INR', 3.500, FALSE, FALSE, 'Online', '2024-01-10', 7, 'active'),
(6, 'IN001000117', 'Fixed Deposit', 100000.00, 'INR', 5.000, FALSE, FALSE, 'Online', '2024-02-20', 7, 'active'),
(8, 'IN001000104', 'Savings', 100000.00, 'INR', 4.000, TRUE, TRUE, 'Online', '2022-07-20', 2, 'active'),
(10, 'IN001000105', 'Current', 7500.00, 'INR', 0.000, FALSE, TRUE, 'Cheque', '2023-09-05', 5, 'active'),
(12, 'IN001000106', 'Savings', 56000.00, 'INR', 3.750, TRUE, FALSE, 'Online', '2024-03-12', 8, 'active'),
(14, 'IN001000107', 'Current', 15000.00, 'INR', 0.000, TRUE, TRUE, 'Cheque', '2023-12-01', 4, 'active'),
(2, 'IN001000108', 'Savings', 87000.00, 'INR', 4.000, TRUE, FALSE, 'Online', '2022-08-25', 6, 'active'),
(2, 'IN001000118', 'Current', 20000.00, 'INR', 0.000, TRUE, TRUE, 'Cheque', '2024-01-10', 1, 'active'),
(3, 'IN001000109', 'Savings', 32000.50, 'INR', 3.500, FALSE, FALSE, 'Online', '2024-02-15', 1, 'active'),
(5, 'IN001000110', 'Current', 4500.00, 'INR', 0.000, TRUE, TRUE, 'Cheque', '2023-10-10', 3, 'active'),
(9, 'IN001000111', 'Savings', 60000.00, 'INR', 3.750, TRUE, FALSE, 'Online', '2023-06-05', 7, 'active'),
(7, 'IN001000112', 'Current', 22000.00, 'INR', 0.000, TRUE, TRUE, 'Cheque', '2022-12-15', 2, 'active'),
(11, 'IN001000113', 'Savings', 41000.00, 'INR', 4.000, TRUE, FALSE, 'Online', '2023-03-20', 5, 'active'),
(13, 'IN001000114', 'Current', 13000.00, 'INR', 0.000, FALSE, TRUE, 'Cheque', '2023-08-30', 8, 'active'),
(15, 'IN001000115', 'Savings', 75000.00, 'INR', 3.500, TRUE, FALSE, 'Online', '2024-01-05', 4, 'active');



CREATE TABLE loan (
    loan_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    customer_id INT UNSIGNED NOT NULL,
    account_id INT UNSIGNED NULL,
    staff_id INT UNSIGNED NULL,
    loan_type VARCHAR(50),
    principal_amount DECIMAL(15,2),
    interest_rate DECIMAL(5,3),
    emi_amount DECIMAL(15,2),
    remaining_principal DECIMAL(15,2),
    start_date DATE,
    end_date DATE,
    nominee_customer_id INT UNSIGNED NULL,
    nominee_name VARCHAR(150),
    nominee_relation VARCHAR(50),
    collateral_description TEXT,
    collateral_value DECIMAL(15,2),
    status VARCHAR(20) DEFAULT 'ongoing',

    CONSTRAINT fk_loan_customer FOREIGN KEY (customer_id) REFERENCES customer(customer_id),
    CONSTRAINT fk_loan_account FOREIGN KEY (account_id) REFERENCES account(account_id),
    CONSTRAINT fk_loan_staff FOREIGN KEY (staff_id) REFERENCES staff(staff_id),
    CONSTRAINT fk_loan_nominee FOREIGN KEY (nominee_customer_id) REFERENCES customer(customer_id)
) ;

-- SELECT staff_id, first_name, last_name FROM staff;

INSERT INTO loan 
(customer_id, account_id, staff_id, loan_type, principal_amount, interest_rate, emi_amount, remaining_principal, start_date, end_date, nominee_customer_id, nominee_name, nominee_relation, collateral_description, collateral_value, status)
VALUES
(1, 1, 1, 'Home', 5000000.00, 7.250, 35000.00, 4800000.00, '2025-01-01', '2035-01-01', NULL, 'Ravi Kumar', 'Friend', 'Flat at MG Road', 3500000.00, 'ongoing'),
(4, 2, 4, 'Car', 1200000.00, 8.500, 30000.00, 1150000.00, '2024-05-01', '2029-05-01', 5, NULL, NULL, 'Car - Hyundai Creta', 800000.00, 'ongoing'),
(6, 3, 6, 'Education', 800000.00, 6.750, 15000.00, 750000.00, '2024-07-10', '2028-07-10', NULL, 'Suresh Rao', 'Uncle', 'Collateral: Bank FD', 500000.00, 'ongoing'),
(8, 4, 8, 'Personal', 300000.00, 12.000, 10000.00, 250000.00, '2025-03-01', '2027-03-01', 2, NULL, NULL, 'Gold Jewelry', 200000.00, 'ongoing'),
(10, 5, 10, 'Home', 6000000.00, 7.000, 45000.00, 5900000.00, '2023-10-15', '2033-10-15', NULL, 'Priya Menon', 'Sister', 'Flat at Residency Road', 4000000.00, 'ongoing'),
(12, 6, 12, 'Car', 900000.00, 8.000, 20000.00, 880000.00, '2024-01-20', '2029-01-20', 6, NULL, NULL, 'Car - Honda City', 600000.00, 'ongoing'),
(14, 7, 14, 'Education', 400000.00, 6.500, 12000.00, 380000.00, '2025-02-05', '2028-02-05', NULL, 'Anjali Patel', 'Cousin', 'Collateral: Bank FD', 250000.00, 'ongoing'),
(2, 8, 2, 'Personal', 250000.00, 11.000, 9000.00, 220000.00, '2024-09-01', '2026-09-01', 3, NULL, NULL, 'Gold Jewelry', 150000.00, 'ongoing'),
(3, 9, 3, 'Home', 3500000.00, 7.250, 25000.00, 3400000.00, '2023-12-10', '2033-12-10', NULL, 'Manish Mehta', 'Friend', 'Flat at Park Street', 2200000.00, 'ongoing'),
(5, 10, 5, 'Car', 700000.00, 8.250, 18000.00, 680000.00, '2025-04-01', '2030-04-01', 1, NULL, NULL, 'Car - Toyota Corolla', 500000.00, 'ongoing');



CREATE TABLE locker (
    locker_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    locker_num VARCHAR(20) UNIQUE NOT NULL,
    customer_id INT UNSIGNED NULL,
    branch_id INT UNSIGNED NULL,
    locker_type VARCHAR(20),
    rent_amount DECIMAL(15,2),
    allocated_date DATE,
    expiry_date DATE,
    status VARCHAR(20) DEFAULT 'vacant',

    CONSTRAINT fk_locker_customer FOREIGN KEY (customer_id) REFERENCES customer(customer_id),
    CONSTRAINT fk_locker_branch FOREIGN KEY (branch_id) REFERENCES branch(branch_id)
);


INSERT INTO locker 
(locker_num, customer_id, branch_id, locker_type, rent_amount, allocated_date, expiry_date, status)
VALUES
('L-001', 1, 1, 'Small', 1200.00, '2024-03-01', '2025-03-01', 'active'),
('L-002', 2, 2, 'Medium', 1800.00, '2024-04-01', '2025-04-01', 'vacant'),
('L-003', 3, 3, 'Large', 2500.00, '2024-05-01', '2025-05-01', 'active'),
('L-004', 4, 4, 'Small', 1200.00, '2024-03-15', '2025-03-15', 'maintenance'),
('L-005', 5, 5, 'Medium', 1800.00, '2024-06-01', '2025-06-01', 'active'),
('L-006', 6, 6, 'Large', 2500.00, '2024-07-01', '2025-07-01', 'vacant'),
('L-007', 7, 7, 'Small', 1200.00, '2024-08-01', '2025-08-01', 'active'),
('L-008', 8, 8, 'Medium', 1800.00, '2024-09-01', '2025-09-01', 'maintenance'),
('L-009', 9, 1, 'Large', 2500.00, '2024-10-01', '2025-10-01', 'active'),
('L-010', 10, 2, 'Small', 1200.00, '2024-11-01', '2025-11-01', 'vacant');




CREATE TABLE investment (
    investment_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    customer_id INT UNSIGNED NOT NULL,
    account_id INT UNSIGNED NULL,
    investment_type VARCHAR(50),
    amount DECIMAL(15,2),
    interest_rate DECIMAL(7,4),
    start_date DATE,
    maturity_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    notes TEXT,
    
    CONSTRAINT fk_investment_customer FOREIGN KEY (customer_id) REFERENCES customer(customer_id),
    CONSTRAINT fk_investment_account FOREIGN KEY (account_id) REFERENCES account(account_id)
);




INSERT INTO investment
(customer_id, account_id, investment_type, amount, interest_rate, start_date, maturity_date, status, notes)
VALUES
(1, 1, 'FD', 50000.00, 5.5000, '2024-11-01', '2025-11-01', 'active', 'Auto renew'),
(2, 2, 'RD', 10000.00, 6.0000, '2024-05-01', '2025-05-01', 'active', 'Monthly deposit'),
(3, 3, 'Mutual Fund', 75000.00, 8.0000, '2024-06-15', '2025-06-15', 'active', 'SIP plan'),
(4, 4, 'Bond', 120000.00, 7.2500, '2024-03-01', '2027-03-01', 'active', '5-year government bond'),
(5, 5, 'FD', 60000.00, 5.7500, '2024-07-01', '2025-07-01', 'active', 'Quarterly interest payout'),
(6, 6, 'RD', 15000.00, 6.2500, '2024-08-01', '2025-08-01', 'active', 'Monthly deposit'),
(7, 7, 'Mutual Fund', 90000.00, 8.5000, '2024-09-01', '2025-09-01', 'active', 'SIP plan'),
(8, 8, 'Bond', 200000.00, 7.0000, '2024-10-01', '2029-10-01', 'active', 'Government bond 5 years'),
(9, 9, 'FD', 80000.00, 5.8000, '2024-11-15', '2025-11-15', 'active', 'Auto renew'),
(10, 10, 'RD', 12000.00, 6.1000, '2024-12-01', '2025-12-01', 'active', 'Monthly deposit');


-- select * from customer;
-- select * from staff;
-- select * from sections;
-- select * from branch;
-- select * from account;
-- select * from loan;
-- select * from locker;
-- select * from investment;


-- DESCRIBE customer;
-- DESCRIBE staff;
-- DESCRIBE account;
-- DESCRIBE investment;
-- DESCRIBE sections;
-- DESCRIBE branch;
-- DESCRIBE loan;
-- DESCRIBE locker;


-- SELECT * FROM account WHERE customer_id NOT IN (SELECT customer_id FROM customer);

