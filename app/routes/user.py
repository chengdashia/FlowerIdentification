from flask import Blueprint, request, jsonify, make_response
from flask_restx import Resource, Namespace, fields
from app import db, api
from app.models.user import User

user_bp = Blueprint('user', __name__)
user_ns = Namespace('user', description='Authentication related operations')

# Define the models for Swagger documentation
register_model = user_ns.model('Register', {
    'username': fields.String(required=True, description='用户名'),
    'password': fields.String(required=True, description='用户密码')
})

login_model = user_ns.model('Login', {
    'username': fields.String(required=True, description='用户名'),
    'password': fields.String(required=True, description='用户密码')
})

update_user_model = user_ns.model('UpdateUser', {
    'user_id': fields.Integer(required=True, description='用户ID'),
    'username': fields.String(required=False, description='用户名'),
    'password': fields.String(required=False, description='密码'),
    'avatar': fields.String(required=False, description='头像(base64格式)')
})


@user_ns.route('/register', methods=['POST'])
class Register(Resource):
    @user_ns.doc(description='用户注册')
    @user_ns.expect(register_model)
    def post(self):
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')

            if User.query.filter_by(username=username).first() is not None:
                return make_response(jsonify({
                    "code": 400,
                    "message": "User already exists"
                }), 400)

            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            return make_response(jsonify({
                "code": 200,
                "message": "User registered successfully"
            }), 200)
        except Exception as e:
            print(f"Error: {str(e)}")
            return make_response(jsonify({
                "code": 500,
                "message": f"Error: {str(e)}"
            }), 500)


@user_ns.route('/login', methods=['POST'])
class Login(Resource):
    @user_ns.doc(description='登录')
    @user_ns.expect(login_model)
    def post(self):
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')

            user = User.query.filter_by(username=username).first()

            if user is None or not user.check_password(password):
                return make_response(jsonify({
                    "code": 401,
                    "message": "Invalid account or password"
                }), 401)

            # 更新用户的登录时间
            from datetime import datetime
            user.updated_time = datetime.now()
            db.session.commit()

            return make_response(jsonify({
                "code": 200,
                "message": "Login successful",
                "data": {
                    "id": user.id,
                    "username": user.username
                }
            }), 200)
        except Exception as e:
            print(f"Error: {str(e)}")
            return make_response(jsonify({
                "code": 500,
                "message": f"Error: {str(e)}"
            }), 500)


@user_ns.route('/info', methods=['POST'])
class UserInfo(Resource):
    @user_ns.doc(description='获取用户信息')
    def post(self):
        user_id = request.headers.get('token')

        if not user_id:
            return make_response(jsonify({
                "code": 401,
                "message": "User ID is required"
            }), 400)

        try:
            user = User.query.get(user_id)

            if user is None:
                return make_response(jsonify({
                    "code": 404,
                    "message": "User not found"
                }), 404)

            return make_response(jsonify({
                "code": 200,
                "message": "Success",
                "data": {
                    "username": user.username,
                    "avatar": user.avatar,  # base64格式的头像
                    "created_time": user.created_time.strftime("%Y-%m-%d %H:%M:%S") if user.created_time else None,
                    "updated_time": user.updated_time.strftime("%Y-%m-%d %H:%M:%S") if user.updated_time else None
                }
            }), 200)
        except Exception as e:
            print(f"Error: {str(e)}")
            return make_response(jsonify({"message": f"Error: {str(e)}"}), 500)


@user_ns.route('/update-info', methods=['POST'])
class UpdateUserInfo(Resource):
    @user_ns.doc(description='修改用户信息')
    @user_ns.expect(update_user_model)
    def post(self):
        try:

            user_id = request.headers.get('token')
            data = request.get_json()
            if not user_id:
                return make_response(jsonify({
                    "code": 400,
                    "message": "User ID is required"
                }), 400)
            user = User.query.get(user_id)
            if user is None:
                return make_response(jsonify({
                    "code": 404,
                    "message": "User not found"
                }), 404)

            # 更新用户信息
            if 'username' in data and data['username']:
                # 检查用户名是否已存在
                existing_user = User.query.filter(User.username == data['username'], User.id != user_id).first()
                if existing_user:
                    return make_response(jsonify({
                        "code": 404,
                        "message": "Username already exists"
                    }), 404)
                user.username = data['username']

            if 'avatar' in data and data['avatar']:
                user.avatar = data['avatar']

            db.session.commit()

            return make_response(jsonify({
                "code": 200,
                "message": "修改成功！！！！"
            }), 200)
        except Exception as e:
            print(f"Error: {str(e)}")
            return make_response(jsonify({"message": f"Error: {str(e)}"}), 500)


@user_ns.route('/update-password', methods=['POST'])
class UpdateUserPassword(Resource):
    @user_ns.doc(description='修改用户密码')
    def post(self):
        try:
            user_id = request.headers.get('token')
            data = request.get_json()
            if not user_id:
                return make_response(jsonify({
                    "code": 400,
                    "message": "User ID is required"
                }), 400)

            user = User.query.get(user_id)

            if user is None:
                return make_response(jsonify({
                    "code": 404,
                    "message": "User not found"
                }), 404)

            # 验证原密码
            if 'old_password' not in data or not data['old_password']:
                return make_response(jsonify({
                    "code": 400,
                    "message": "Old password is required"
                }), 400)

            if not user.check_password(data['old_password']):
                return make_response(jsonify({
                    "code": 404,
                    "message": "原密码不正确"
                }), 404)

            # 验证新密码
            if 'new_password' not in data or not data['new_password']:
                return make_response(jsonify({
                    "code": 400,
                    "message": "New password is required"
                }), 400)

            # 更新密码
            user.set_password(data['new_password'])

            db.session.commit()

            return make_response(jsonify({
                "code": 200,
                "message": "密码修改成功！"
            }), 200)
        except Exception as e:
            print(f"Error: {str(e)}")
            return make_response(jsonify({"message": f"Error: {str(e)}"}), 500)


# Add the namespace to the api
api.add_namespace(user_ns)
