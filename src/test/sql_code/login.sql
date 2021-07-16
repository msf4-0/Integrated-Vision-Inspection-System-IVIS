-- >>>> New Session Log at Log-IN >>>>
INSERT INTO session_log (
    user_id)
VALUES (
    % s)
RETURNING
    id;

-- this state would include id, user_id, login_at
-- >>>> Update Session Log at Log-OUT >>>>
UPDATE
    session_log
SET
    logout_at = % s --here substitute current datetime values
WHERE
    id = % s -- Current session_id
RETURNING
    *;

-- this state would include id, user_id, login_at,logout_at (COMPLETE)
-- >>>> Update Account Status
UPDATE
    USER
SET
    status = % s -- substitute account status
WHERE
    id = % s;

-- user_id
-- UPDATE TIME
CREATE OR REPLACE FUNCTION update_modified_column ()
    RETURNS TRIGGER
    AS $$
BEGIN
    NEW.modified = now();
    RETURN NEW;
END;
$$
LANGUAGE 'plpgsql';

CREATE TRIGGER update_customer_modtime
    BEFORE UPDATE ON customer
    FOR EACH ROW
    EXECUTE PROCEDURE update_modified_column ();

-- Secondary
CREATE OR REPLACE FUNCTION trigger_update_timestamp ()
    RETURNS TRIGGER
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$
LANGUAGE plpgsql;

CREATE TRIGGER set_timestamp
    BEFORE UPDATE ON todos
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

