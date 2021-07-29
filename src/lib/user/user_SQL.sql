-- Query User details
SELECT
    emp_id,
    username,
    first_name,
    last_name,
    email,
    department,
    position,
    (
        SELECT
            r.name
        FROM
            public.roles r
        WHERE
            r.id = roles_id) AS "Role",
    (
        SELECT
            x.name
        FROM
            public.account_status x
        WHERE
            x.id = status_id) AS "Account Status"
FROM
    public.users
WHERE
    id = % s;

-- Update Last Activity
UPDATE
    public.users
SET
    last_activity = datetime.now().astimezone ()
WHERE
    id = % s;

