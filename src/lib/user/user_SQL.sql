-- Query User details
SELECT
    emp_id
    , username
    , first_name
    , last_name
    , email
    , department
    , position
    , (
        SELECT
            r.name
        FROM
            public.roles r
        WHERE
            r.id = roles_id) AS "Role"
    , (
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


/* Insert User  */
INSERT INTO public.users (
    emp_id
    , first_name
    , last_name
    , email
    , department
    , position
    , username
    , roles_id
    , psd)
VALUES (
    % s
    , % s
    , % s
    , % s
    , % s
    , % s
    , % s
    , (
        SELECT
            r.id
        FROM
            public.roles r
        WHERE
            r.name = % s) , % s)
RETURNING
    id;

