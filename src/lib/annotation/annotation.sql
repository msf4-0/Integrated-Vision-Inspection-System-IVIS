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
            % s = % s);

