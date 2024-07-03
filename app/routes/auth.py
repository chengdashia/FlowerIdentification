from flask import Blueprint, request, jsonify, make_response
from flask_restx import Resource, Namespace, fields
from app import db, api
from app.models.user import User

auth_bp = Blueprint('auth', __name__)
auth_ns = Namespace('auth', description='Authentication related operations')

# Define the models for Swagger documentation
register_model = auth_ns.model('Register', {
    'username': fields.String(required=True, description='用户名'),
    'account': fields.String(required=True, description='用户账号'),
    'password': fields.String(required=True, description='用户密码')
})

login_model = auth_ns.model('Login', {
    'account': fields.String(required=True, description='用户账号'),
    'password': fields.String(required=True, description='用户密码')
})


@auth_ns.route('/register')
class Register(Resource):
    @auth_ns.doc(description='用户注册')
    @auth_ns.expect(register_model)
    def post(self):
        data = request.get_json()
        username = data.get('username')
        account = data.get('account')
        password = data.get('password')

        if User.query.filter_by(account=account).first() is not None:
            return make_response(jsonify({"message": "Account already exists"}), 400)

        new_user = User(username=username, account=account)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        return make_response(jsonify({"message": "User registered successfully"}), 201)


@auth_ns.route('/login')
class Login(Resource):
    @auth_ns.doc(description='登录')
    @auth_ns.expect(login_model)
    def post(self):
        data = request.get_json()
        account = data.get('account')
        password = data.get('password')

        user = User.query.filter_by(account=account).first()

        if user is None or not user.check_password(password):
            return make_response(jsonify({"message": "Invalid account or password"}), 401)

        return make_response(jsonify({
            "message": "Login successful",
            "user": {
                "id": user.id,
                "username": user.username,
                "account": user.account
            }
        }), 200)


# Add the namespace to the api
api.add_namespace(auth_ns)
