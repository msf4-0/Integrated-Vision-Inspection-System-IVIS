/* Query model table */
-- used at Main Model Page
SELECT
    m.id AS "ID"
    , m.name AS "Name"
    , (
        SELECT
            f.name AS "Framework"
        FROM
            public.framework f
        WHERE
            f.id = m.framework_id) , (
        SELECT
            mt.name AS "Model Type"
        FROM
            public.model_type mt
        WHERE
            mt.id = m.model_type_id) , (
        SELECT
            dt.name AS "Deployment Type"
        FROM
            public.deployment_type dt
        WHERE
            dt.id = m.deployment_id) , (
        /* Replace NULL with '-' */
        SELECT
            CASE WHEN m.training_id IS NULL THEN
                '-'
            ELSE
                (
                    SELECT
                        t.name
                    FROM
                        public.training t
                    WHERE
                        t.id = m.training_id)
            END AS "Training Name")
    , m.description AS "Description"
    , m.metrics AS "Metrics"
    , m.model_path AS "Model Path"
FROM
    public.models m
ORDER BY
    ID ASC;


/* To query based on deployment type */
-- used when creating new training / at project sub pages
SELECT
    m.id AS "ID"
    , m.name AS "Name"
    , (
        SELECT
            f.name AS "Framework"
        FROM
            public.framework f
        WHERE
            f.id = m.framework_id) , (
        SELECT
            mt.name AS "Model Type"
        FROM
            public.model_type mt
        WHERE
            mt.id = m.model_type_id) , (
        /* Replace NULL with '-' */
        SELECT
            CASE WHEN m.training_id IS NULL THEN
                '-'
            ELSE
                (
                    SELECT
                        t.name
                    FROM
                        public.training t
                    WHERE
                        t.id = m.training_id)
            END AS "Training Name") ,
    --        m.updated_at  AS "Date/Time",
    m.description AS "Description"
    , m.metrics AS "Metrics"
    , m.model_path AS "Model Path"
FROM
    public.models m
    INNER JOIN public.deployment_type dt ON dt.name = 'Object Detection with Bounding Boxes'
ORDER BY
    m.id ASC;


/* Query all fields */
SELECT
    m.id AS "ID"
    , m.name AS "Name"
    , (
        SELECT
            f.name AS "Framework"
        FROM
            public.framework f
        WHERE
            f.id = m.framework_id) , (
        SELECT
            mt.name AS "Model Type"
        FROM
            public.model_type mt
        WHERE
            mt.id = m.model_type_id) , (
        SELECT
            dt.name AS "Deployment Type"
        FROM
            public.deployment_type dt
        WHERE
            dt.id = m.deployment_id) , (
        /* Replace NULL with '-' */
        SELECT
            CASE WHEN m.training_id IS NULL THEN
                '-'
            ELSE
                (
                    SELECT
                        t.name
                    FROM
                        public.training t
                    WHERE
                        t.id = m.training_id)
            END AS "Training Name")
    , m.updated_at AS "Date/Time"
    , m.description AS "Description"
    , m.metrics AS "Metrics"
    , m.model_path AS "Model Path"
FROM
    public.models m
WHERE
    m.id = % s;


/*
 Create New Model at DB
Need to support both 'User Upload Model' and 'Project Models' 
 */
INSERT INTO public.models (
    name
    , description
    , metrics
    , model_path
    , model_type_id
    , framework_id
    , deployment_id
    , training_id)
VALUES (
    % s
    , % s
    , % s::jsonb
    , % s
    , (
        SELECT
            mt.id
        FROM
            public.model_type mt
        WHERE
            mt.name = % s) , (
            SELECT
                f.id
            FROM
                public.framework f
            WHERE
                f.name = % s) , (
                SELECT
                    dt.id
                FROM
                    public.deployment_type dt
                WHERE
                    dt.name = % s) , % s)
    RETURNING
        id;


/* Update Model Table */
UPDATE
    public.models
SET
    name = % s
    , description = % s
    , metrics = % s::jsonb
    , model_path = % s
    , model_type_id = (
        SELECT
            mt.id
        FROM
            public.model_type mt
        WHERE
            mt.name = % s) , framework_id = (
        SELECT
            f.id
        FROM
            public.framework f
        WHERE
            f.name = % s) , deployment_id = (
        SELECT
            dt.id
        FROM
            public.deployment_type dt
        WHERE
            dt.name = % s) , training_id = % s
WHERE
    id = % s
RETURNING
    id;

