-- Query Deployment_ID from table
SELECT
    id
FROM
    public.deployment_type
WHERE
    name = % s;

-- Query Dataset List from dataset table
SELECT
    id,
    name,
    dataset_size,
    updated_at
FROM
    public.dataset;

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

