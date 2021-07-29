-- NEW ANNOTATION
-- Insert new annotations into Database
-- returns annotation_id for task
INSERT INTO public.annotations (
    results,
    project_id,
    users_id,
    task_id)
VALUES (
    % s ::jsonb,
    % s,
    % s,
    % s) --results , project_id, user_id and task_id
RETURNING
    id;

-- Update task table annotation_id,is_labelled, skipped, annotation_type_id
-- skipped annotation"type_id
UPDATE
    public.task
SET
    (annotation_id = % s),
    (is_labelled = % s),
    (skipped = % s)
WHERE
    id = % s;

-- UPDATE ANNOTATION
-- Update annotations table
UPDATE
    public.annotations
SET
    (results = % s::jsonb),
    (users_id = % s)
WHERE
    id = % s
RETURNING
    *;

--SKIPPING TASK
-- Update task table skipped
UPDATE
    public.task
SET
    (skipped = % s)
WHERE
    id = % s;

-- DELETE ANNOTATION
-- Delete annotations from annotations table
DELETE FROM public.annotation
WHERE id = % s
RETURNING
    *;

-- Check if Task exists
SELECT
    EXISTS (
        SELECT
            % s
        FROM
            {}
        WHERE
            name = % s
            AND project_id = % s
            AND dataset_id = % s);

-- Insert Image into Task (Create Task)
INSERT INTO public.task (
    name,
    project_id,
    dataset_id)
VALUES (
    % s,
    % s,
    % s)
RETURNING
    id;

-- Query Task details
SELECT
    id,
    is_labelled,
    skipped
FROM
    public.task
WHERE
    name = % s
    AND project_id = % s
    AND dataset_id = % s;

-- Query Annotations
SELECT
    id,
    result
FROM
    public.annotations
WHERE
    task_id = % s;

