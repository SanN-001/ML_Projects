class Employee:
    def __init__(self, emp_id, name, role):
        self.emp_id = emp_id
        self.name = name
        self.role = role

class AVLNode:
    def __init__(self, employee):
        self.employee = employee
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def get_height(self, node):
        return node.height if node else 0

    def get_balance(self, node):
        return self.get_height(node.left) - self.get_height(node.right) if node else 0

    def rotate_right(self, z):
        y = z.left
        T2 = y.right
        y.right = z
        z.left = T2
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    def rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    def insert(self, node, employee):
        if not node:
            return AVLNode(employee)

        if employee.emp_id < node.employee.emp_id:
            node.left = self.insert(node.left, employee)
        else:
            node.right = self.insert(node.right, employee)

        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        balance = self.get_balance(node)

        # Left Heavy Case
        if balance > 1:
            if employee.emp_id < node.left.employee.emp_id:
                return self.rotate_right(node)
            else:
                node.left = self.rotate_left(node.left)
                return self.rotate_right(node)

        # Right Heavy Case
        if balance < -1:
            if employee.emp_id > node.right.employee.emp_id:
                return self.rotate_left(node)
            else:
                node.right = self.rotate_right(node.right)
                return self.rotate_left(node)

        return node

    def min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current

    def delete(self, node, emp_id):
        if not node:
            return node

        if emp_id < node.employee.emp_id:
            node.left = self.delete(node.left, emp_id)
        elif emp_id > node.employee.emp_id:
            node.right = self.delete(node.right, emp_id)
        else:
            if not node.left:
                return node.right
            elif not node.right:
                return node.left

            temp = self.min_value_node(node.right)
            node.employee = temp.employee
            node.right = self.delete(node.right, temp.employee.emp_id)

        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        balance = self.get_balance(node)

        # Left Heavy Case
        if balance > 1:
            if self.get_balance(node.left) >= 0:
                return self.rotate_right(node)
            else:
                node.left = self.rotate_left(node.left)
                return self.rotate_right(node)

        # Right Heavy Case
        if balance < -1:
            if self.get_balance(node.right) <= 0:
                return self.rotate_left(node)
            else:
                node.right = self.rotate_right(node.right)
                return self.rotate_left(node)

        return node

    def search(self, node, emp_id):
        if not node or node.employee.emp_id == emp_id:
            return node
        if emp_id < node.employee.emp_id:
            return self.search(node.left, emp_id)
        return self.search(node.right, emp_id)

    def pre_order(self, node):
        if not node:
            return
        print(f"ID: {node.employee.emp_id}, Name: {node.employee.name}, Role: {node.employee.role}")
        self.pre_order(node.left)
        self.pre_order(node.right)

# Example usage
if __name__ == "__main__":
    tree = AVLTree()
    root = None

    employees = [
        Employee(10, "Alice", "Manager"),
        Employee(20, "Bob", "Developer"),
        Employee(30, "Charlie", "Designer"),
        Employee(40, "David", "Engineer"),
        Employee(50, "Eve", "Analyst")
    ]

    # Insert employees
    for emp in employees:
        root = tree.insert(root, emp)

    print("AVL Tree in Pre-order:")
    tree.pre_order(root)

    # Search for an employee
    found_emp = tree.search(root, 30)
    if found_emp:
        print(f"\nFound Employee: {found_emp.employee.name}, Role: {found_emp.employee.role}")
    else:
        print("\nEmployee not found")

    # Delete employee with ID 20
    root = tree.delete(root, 20)
    print("\nPre-order after deletion of Employee with ID 20:")
    tree.pre_order(root)
