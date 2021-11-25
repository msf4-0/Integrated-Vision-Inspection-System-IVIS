from enum import IntEnum

from core.utils.form_manager import reset_page_attributes

MAIN_PAGE_OPTIONS = ("Login", "Logout", "Projects", "User Management")


class MainPagination(IntEnum):
    CreateUser = 0
    Login = 1
    Projects = 2
    UserManagement = 3
    UserInfo = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return MainPagination[s]
        except KeyError:
            raise ValueError()

    @property
    def display_name(self) -> str:
        return MAIN_PAGE_OPTIONS[self.value]

    @staticmethod
    def get_enum_from_display_name(display_name: str) -> IntEnum:
        for enum_obj in MainPagination:
            if enum_obj.display_name == display_name:
                return enum_obj
        else:
            raise ValueError(
                f"'{display_name}' is not a valid full name for user roles.")


class UserManagementPagination(IntEnum):
    Dashboard = 0
    CreateUser = 1
    EditUser = 2

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return UserManagementPagination[s]
        except KeyError:
            raise ValueError()


def reset_user_management_page():
    user_manage_attributes = ["user_manage_pagination", "current_user"]

    reset_page_attributes(user_manage_attributes)
