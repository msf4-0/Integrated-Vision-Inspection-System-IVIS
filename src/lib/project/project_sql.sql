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
NSERT INTO public.project (name,
    description,
    project_path,
    deployment_id)
VALUES (% s,
    % s,
    % s,
    % s)
RETURNING
    id;

-- Insert to project_dataset table
INSERT INTO public.project_dataset (
    project_id,
    dataset_id)
VALUES (
    % s,
    % s);

-- Insert Editor Base Config into Editor Table
INSERT INTO public.editor (
    name,
    editor_config,
    project_id)
VALUES (
    % s,
    % s,
    % s)
RETURNING
    id;

-- Query all field of Project
SELECT
    p.id,
    p.name,
    description,
    dt.name AS deployment_type,
    deployment_id,
    project_path
FROM
    public.project p
    LEFT JOIN deployment_type dt ON dt.id = p.deployment_id
WHERE
    p.id = % s;

-- Query all project
SELECT
    p.id,
    p.name,
    description,
    dt.name deployment_type,
    deployment_id,
    project_path
FROM
    public.project p
    LEFT JOIN deployment_type dt ON dt.id = p.deployment_id;

