-- Insert new annotations into Database
-- returns annotation_id for task
INSERT INTO public.annotations results
VALUES (
    % s ::jsonb,
    % s) --results and user_id
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

-- Update annotations table
-- skipped annotation"type_id
UPDATE
    public.annotations
SET
    (results = % s::jsonb)
WHERE
    id = % s;

