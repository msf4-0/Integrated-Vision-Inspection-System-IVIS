-- Query Deployment Type List
SELECT
    name
FROM
    deployment_type
ORDER BY
    id ASC;

-- Insert random training data for testing
INSERT INTO public.training (
    name,
    pre_trained_model_id,
    framework_id,
    project_id)
VALUES (
    'not so serious training',
    1,
    1,
    7);

-- Insert random model data for testing
INSERT INTO public.models (
    name,
    training_id,
    framework_id)
VALUES (
    'ssd300 retinanet merdeka model',
    7,
    1);

-- get Project Model Path
SELECT
    p.project_path,
    t.name
FROM
    public.models m
    INNER JOIN public.training t ON m.training_id = t.id
    INNER JOIN public.project p ON t.project_id = p.id
WHERE
    m.id = % s;

SELECT
    % s
FROM
    % s;
-- query model based on deployment
SELECT
    *
FROM
    public.pre_trained_models pt
WHERE
    deployment_id = (
        SELECT
            dt.id
        FROM
            public.deployment_type dt
        WHERE
            dt.name = 'Object Detection with Bounding Boxes');

