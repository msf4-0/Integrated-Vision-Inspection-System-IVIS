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

-- Query PROJECT_DATASET table
SELECT
    d.id AS dataset_id,
    d.name AS dataset_name,
    d.dataset_size,
    pd.updated_at
FROM
    public.project_dataset pd
    LEFT JOIN public.dataset d ON d.id = pd.dataset_id
WHERE
    pd.project_id = % s;

-- Query PRE_TRAINED_MODELS table
SELECT
    pt.id AS "ID",
    pt.name AS "Name",
    f.name AS "Framework",
    dt.name AS "Deployment Type",
    pt.model_path AS "Model Path"
FROM
    public.pre_trained_models pt
    LEFT JOIN public.framework f ON f.id = pt.framework_id
    LEFT JOIN public.deployment_type dt ON dt.id = pt.deployment_id;

-- Queyr list of Framework
SELECT
    id,
    name
FROM
    public.framework;

-- Insert into Training Table
INSERT INTO public.training (
    name,
    description,
    training_param,
    augmentation,
    pre_trained_model_id,
    framework_id,
    project_id,
    partition_size)
VALUES (
    % s,
    % s,
    % s,
    % s,
    (
        SELECT
            pt.id
        FROM
            public.pre_trained_models pt
        WHERE
            pt.name = % s), (
            SELECT
                f.id
            FROM
                public.framework f
            WHERE
                f.name = % s), % s, % s)
RETURNING
    id;

-- Insert into Project_Training table
INSERT INTO public.project_training (
    project_id,
    training_id)
VALUES (
    % s,
    % s);

-- Insert into Model table
INSERT INTO public.models (
    name,
    model_path,
    training_id,
    framework_id,
    deployment_id)
VALUES (
    % s,
    % s,
    % s,
    (
        SELECT
            f.id
        FROM
            public.framework f
        WHERE
            f.name = % s), (
            SELECT
                dt.id
            FROM
                public.deployment_type dt
            WHERE
                dt.name = % s))
RETURNING
    id;

--Insert into training_dataset table
INSERT INTO public.training_dataset (
    training_id,
    dataset_id)
VALUES (
    % s,
    (
        SELECT
            id
        FROM
            public.dataset d
        WHERE
            d.name = % s))
-- Query Editor config
SELECT
    editor_config
FROM
    public.editor
WHERE
    project_id = % s;

-- Update Editor Config
UPDATE
    public.editor
SET
    editor_config = % s
WHERE
    project_id = % s;

