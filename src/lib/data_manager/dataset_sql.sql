-- Query Deployment_ID from table
SELECT
    id
FROM
    public.deployment_type
WHERE
    name = % s;

