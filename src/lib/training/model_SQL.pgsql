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

