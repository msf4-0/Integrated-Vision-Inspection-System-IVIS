/* Query All Project Training */
SELECT
    t.id AS "ID",
    t.name AS "Training Name",
    (
        SELECT
            CASE
                WHEN t.training_model_id IS NULL THEN '-'
                ELSE (
                    SELECT
                        m.name
                    FROM
                        public.models m
                    WHERE
                        m.id = t.training_model_id
                )
            END AS "Model Name"
    ),
    (
        SELECT
            m.name AS "Base Model Name"
        FROM
            public.models m
        WHERE
            m.id = t.attached_model_id
    ),
    t.is_started AS "Is Started",
    CASE
        WHEN t.progress IS NULL THEN '{}'
        ELSE t.progress
    END AS "Progress",
    /*Give empty JSONB if training progress has not start*/
    t.updated_at AS "Date/Time"
FROM
    public.project_training pro_train
    INNER JOIN training t ON t.id = pro_train.training_id
WHERE
    pro_train.project_id = 43;