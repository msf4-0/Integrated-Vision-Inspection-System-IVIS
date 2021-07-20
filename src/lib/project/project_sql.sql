-- Query Deployment_ID from table
SELECT
    id
FROM
    public.deployment_type
WHERE
    name = % s;

--Check if name exists
SELECT
    EXISTS (
        SELECT
            name
        FROM
            public.dataset
        WHERE
            name = % s);

-- Submit Dataset details into public.dataset table
INSERT INTO public.dataset (
    name,
    description,
    file_type,
    dataset_path,
    dataset_size,
    deployment_id)
VALUES (
    % s,
    % s,
    % s,
    % s,
    % s,
    % s)
RETURNING
    id;

