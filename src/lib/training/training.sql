/* Query All Project Training */
SELECT
    t.id AS "ID"
    , t.name AS "Training Name"
    , (
        SELECT
            CASE WHEN t.training_model_id IS NULL THEN
                '-'
            ELSE
                (
                    SELECT
                        m.name
                    FROM
                        public.models m
                    WHERE
                        m.id = t.training_model_id)
            END AS "Model Name")
    , (
        SELECT
            m.name AS "Base Model Name"
        FROM
            public.models m
        WHERE
            m.id = t.attached_model_id) , t.is_started AS "Is Started" , CASE WHEN t.progress IS NULL THEN
        '{}'
    ELSE
        t.progress
    END AS "Progress" ,
    /*Give empty JSONB if training progress has not start*/
    t.updated_at AS "Date/Time"
FROM
    public.project_training pro_train
    INNER JOIN training t ON t.id = pro_train.training_id
WHERE
    pro_train.project_id = 43;


/* Insert Training Info */
INSERT INTO public.training (
    name
    , description
    , partition_size)
VALUES (
    % s
    , % s
    , % s ::jsonb)
ON CONFLICT (
    name)
    DO UPDATE SET
        description = % s
        , partition_size = % s
    RETURNING
        id;


/* Insert into Project Trainign */
INSERT INTO public.project_training (
    project_id
    , training_id)
VALUES (
    % s
    , % s)
ON CONFLICT (
    project_id
    , training_id)
    DO NOTHING;


/* Insert into Training Dataset */
INSERT INTO public.training_dataset (
    training_id
    , dataset_id)
VALUES (
    % s
    , (
        SELECT
            d.id
        FROM
            public.dataset d
        WHERE
            d.name = % s))
ON CONFLICT ON CONSTRAINT training_dataset_pkey
    DO NOTHING;


/* Update Training Table */
UPDATE
    public.training
SET
    name = % s
    , description = % s
    , partition_size = % s
WHERE
    id = % s;


/* Update Training Info */
UPDATE
    public.training
SET
    name = % s
    , description = % s
    , partition_size = % s
WHERE
    id = % s
RETURNING
    id;


/* Delete Training Dataset */
DELETE FROM public.training_dataset
WHERE training_id = % s
    AND dataset_id = (
        SELECT
            id
        FROM
            public.dataset d
        WHERE
            d.name = % s)
RETURNING
    dataset_id;


/* Query all fields Training Table */
SELECT
    id
    , name
    , description
    , training_param
    , augmentation
    , is_started
    , progress
    , partition_size
FROM
    public.training
WHERE
    id = % s;


/* Update Training table with Training and Attached Model ID */
UPDATE
    public.training
SET
    training_model_id = % s
    , attached_model_id = % s
WHERE
    id = % s;

