class UsersController < ApplicationController
  before_action :authenticate_user!
  
  def history
    @history_records = current_user.history_records
  end

end