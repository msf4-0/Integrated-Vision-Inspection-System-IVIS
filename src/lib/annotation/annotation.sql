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
    result,
    (u.id,
        u.email,
        u.first_name,
        u.last_name),
    created_at,
    updated_at
FROM
    public.annotations a
    INNER JOIN public.users u ON a.users_id = u.id
WHERE
    task_id = % s;

-- Query all Task
SELECT
    t.id,
    t.name AS "Task Name",
    (
        SELECT
            CASE WHEN (
                SELECT
                    first_name || ' ' || last_name AS "Created By"
                FROM
                    public.users u
                WHERE
                    u.id = a.users_id) IS NULL THEN
                '-'
            END AS "Created By"),
    (
        SELECT
            d.name AS "Dataset Name"
        FROM
            public.dataset d
        WHERE
            d.id = t.dataset_id), t.is_labelled AS "Is Labelled", t.skipped AS "Skipped", t.updated_at AS "Date/Time"
FROM
    public.task t
    LEFT JOIN public.annotations a ON a.id = t.annotation_id
WHERE
    t.project_id = %s
ORDER BY
    t.id;

